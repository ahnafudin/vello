// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent-axis Gaussian blur on the CPU: each axis runs its own decimation plan with one-axis resampling
//! and a one-axis convolution, optionally in linear-light premultiplied colour.

use super::FilterEffect;
use super::gaussian_blur::{
    convolve_x, convolve_y, downscale_x, downscale_y, upscale_x, upscale_y,
};
use crate::filter::context::ScratchBuffer;
use vello_common::filter::gaussian_blur::{DecimationSizer, GaussianBlur};
use vello_common::filter::gaussian_xy::PreparedGaussianXY;
use vello_common::filter_effects::EdgeMode;
use vello_common::peniko::color::PremulRgba8;
#[cfg(not(feature = "std"))]
use vello_common::peniko::kurbo::common::FloatFuncs as _;
use vello_common::pixmap::Pixmap;

impl FilterEffect for PreparedGaussianXY {
    fn execute_lowp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        if self.x.std_deviation <= 0.0 && self.y.std_deviation <= 0.0 {
            return;
        }
        if self.linear_rgb {
            map_colors(pixmap, srgb_to_linear);
        }
        blur_axis(pixmap, filter_scratch, &self.x, true);
        blur_axis(pixmap, filter_scratch, &self.y, false);
        if self.linear_rgb {
            map_colors(pixmap, linear_to_srgb);
        }
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, filter_scratch: &mut ScratchBuffer) {
        // As for the uniform blur, only the low-precision path exists so far.
        Self::execute_lowp(self, pixmap, filter_scratch);
    }
}

/// One axis of the blur. Upsampling doubles the decimated length, so the blurred axis must be a multiple of
/// `2^n_decimations`, or the last upsampled pixel would land past the row. Any other length is blurred in a copy
/// padded with transparent pixels, which is what `EdgeMode::None` reads past the edge anyway. A pixmap too long
/// to pad within `u16` is left unblurred on that axis.
fn blur_axis(
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    blur: &GaussianBlur,
    horizontal: bool,
) {
    if blur.std_deviation <= 0.0 {
        return;
    }
    let (width, height) = (pixmap.width(), pixmap.height());
    let length = u32::from(if horizontal { width } else { height });
    let step = 1_u32 << blur.n_decimations.min(16);
    let Ok(padded) = u16::try_from(length.div_ceil(step) * step) else {
        return;
    };
    if u32::from(padded) == length {
        blur_axis_in_place(pixmap, filter_scratch, blur, horizontal);
        return;
    }
    let mut work = if horizontal {
        Pixmap::new(padded, height)
    } else {
        Pixmap::new(width, padded)
    };
    copy_region(pixmap, &mut work, width, height);
    blur_axis_in_place(&mut work, filter_scratch, blur, horizontal);
    copy_region(&work, pixmap, width, height);
}

/// Downsamples one axis only, convolves along it, and upsamples it back, in place.
fn blur_axis_in_place(
    pixmap: &mut Pixmap,
    filter_scratch: &mut ScratchBuffer,
    blur: &GaussianBlur,
    horizontal: bool,
) {
    let kernel = &blur.kernel[..usize::from(blur.kernel_size)];
    let radius = (kernel.len() / 2) as u8;
    let edge_mode = EdgeMode::None;
    let scratch = filter_scratch.get_scratch_buffer(pixmap.width(), pixmap.height());
    let mut sizer = DecimationSizer::new(pixmap.width(), pixmap.height());
    for _ in 0..blur.n_decimations {
        let (w, h) = sizer.current();
        if horizontal {
            downscale_x(pixmap, w, h, w.div_ceil(2), edge_mode);
        } else {
            downscale_y(pixmap, w, h, h.div_ceil(2), edge_mode);
        }
        sizer.downscale_axis(horizontal);
    }
    let (w, h) = sizer.current();
    if horizontal {
        convolve_x(pixmap, scratch, w, h, kernel, radius, edge_mode);
    } else {
        convolve_y(pixmap, scratch, w, h, kernel, radius, edge_mode);
    }
    copy_region(scratch, pixmap, w, h);
    for _ in 0..blur.n_decimations {
        let (w, h) = sizer.current();
        if horizontal {
            upscale_x(pixmap, w, h, edge_mode);
        } else {
            upscale_y(pixmap, w, h, edge_mode);
        }
        sizer.upscale_axis();
    }
}

/// Copies the top-left `width` x `height` pixels of `src` into `dst`.
fn copy_region(src: &Pixmap, dst: &mut Pixmap, width: u16, height: u16) {
    for y in 0..height {
        for x in 0..width {
            dst.set_pixel(x, y, src.sample(x, y));
        }
    }
}
/// Applies `transfer` to the colour of every pixel, keeping premultiplication: unpremultiply, map, premultiply.
fn map_colors(pixmap: &mut Pixmap, transfer: fn(f32) -> f32) {
    for pixel in pixmap.data_mut() {
        if pixel.a == 0 {
            continue;
        }
        let alpha = f32::from(pixel.a) / 255.0;
        let map = |channel: u8| {
            let straight = (f32::from(channel) / 255.0 / alpha).min(1.0);
            (transfer(straight) * alpha * 255.0).round() as u8
        };
        *pixel = PremulRgba8 {
            r: map(pixel.r),
            g: map(pixel.g),
            b: map(pixel.b),
            a: pixel.a,
        };
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::kurbo::Affine;

    fn dot() -> Pixmap {
        let mut pixmap = Pixmap::new(33, 33);
        let opaque = PremulRgba8 {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        pixmap.set_pixel(16, 16, opaque);
        pixmap
    }

    #[test]
    fn one_axis_blur_spreads_along_that_axis_only() {
        // 33 is odd, so every decimated case also takes the padded path.
        for (sigma, horizontal) in [
            ([3.0, 0.0], true),
            ([0.0, 3.0], false),
            ([6.0, 0.0], true),
            ([0.0, 12.0], false),
        ] {
            let mut pixmap = dot();
            let mut scratch = ScratchBuffer::default();
            PreparedGaussianXY::new(sigma, false, &Affine::IDENTITY)
                .execute_lowp(&mut pixmap, &mut scratch);
            let along = if horizontal {
                pixmap.sample(19, 16)
            } else {
                pixmap.sample(16, 19)
            };
            let across = if horizontal {
                pixmap.sample(16, 19)
            } else {
                pixmap.sample(19, 16)
            };
            assert!(along.a > 0, "{sigma:?}: no spread along the blurred axis");
            assert_eq!(across.a, 0, "{sigma:?}: spread across the unblurred axis");
        }
    }

    #[test]
    fn linear_light_round_trip_keeps_opaque_extremes() {
        let mut pixmap = Pixmap::new(2, 1);
        let white = PremulRgba8 {
            r: 255,
            g: 255,
            b: 255,
            a: 255,
        };
        pixmap.set_pixel(0, 0, white);
        map_colors(&mut pixmap, srgb_to_linear);
        map_colors(&mut pixmap, linear_to_srgb);
        assert_eq!(pixmap.sample(0, 0), white);
        assert_eq!(pixmap.sample(1, 0).a, 0);
    }
}
