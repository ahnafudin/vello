// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Independent 2D Gaussian convolution over a rounded rectangle and the entire output pixel.

use super::{integrated_cdf, paint};
use core::f64::consts::SQRT_2;
use core::ops::Range;
use vello_cpu::kurbo::Rect;

const SIZE: (u16, u16) = (256, 192);

// Integrate a horizontal source segment convolved with a Gaussian over [x, x + 1].
fn pixel_span(x: f64, left: f64, right: f64, sigma: f64) -> f64 {
    let from = |edge| integrated_cdf(x + 1.0 - edge, sigma) - integrated_cdf(x - edge, sigma);
    from(left) - from(right)
}

// Integrate horizontal slices of the source. At each slice the Gaussian is separable,
// so both footprint integrals are analytic and only the source ordinate uses quadrature.
fn reference(rect: Rect, r: f64, sigma: f64, xs: Range<u16>, ys: Range<u16>) -> Vec<f64> {
    let width = xs.len();
    let height = ys.len();
    let mut result = vec![0.0; width * height];
    let breaks = [rect.y0, rect.y0 + r, rect.y1 - r, rect.y1];
    for slice in breaks.windows(2).filter(|pair| pair[1] > pair[0]) {
        let length = slice[1] - slice[0];
        // Composite Simpson on each arc/straight interval. The smooth substitution
        // clusters ordinates near the arc ends and has zero endpoint weight.
        const STEPS: u32 = 192;
        for i in 1..STEPS {
            let t = f64::from(i) / f64::from(STEPS);
            let v = slice[0] + length * t * t * (3.0 - 2.0 * t);
            let dy = (rect.y0 + r - v).max(v - (rect.y1 - r)).max(0.0);
            let dx = r - (r * r - dy * dy).max(0.0).sqrt();
            let weight = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
            let dv = weight * length * 6.0 * t * (1.0 - t) / (3.0 * f64::from(STEPS));
            let horizontal: Vec<_> = xs
                .clone()
                .map(|x| pixel_span(f64::from(x), rect.x0 + dx, rect.x1 - dx, sigma))
                .collect();
            for (y, row) in ys.clone().zip(result.chunks_exact_mut(width)) {
                let distance = f64::from(y) - v;
                let erf = |d| libm::erf(d / (SQRT_2 * sigma));
                let vertical = dv * 0.5 * (erf(distance + 1.0) - erf(distance));
                for (value, x_mass) in row.iter_mut().zip(&horizontal) {
                    *value += vertical * x_mass;
                }
            }
        }
    }
    result
}

fn check_shape(bounds: [u16; 4], r: u16, thin: bool) {
    let [x, y, x1, y1] = bounds;
    let rect = Rect::new(f64::from(x), f64::from(y), f64::from(x1), f64::from(y1));
    let (mid_x, mid_y) = (x.midpoint(x1), y.midpoint(y1));
    for blur in [5_u16, 6, 10] {
        let sigma = f32::from(blur) / 2.0;
        let pad = 2 * blur;
        let regions = [
            ("long edge", mid_x - 2..mid_x + 2, y - pad..y + pad, 2.0),
            ("corner", x - pad..x + r + pad, y - pad..y + r + pad, 8.0),
            (
                "short end",
                x - pad..x + pad,
                mid_y - 2..mid_y + 2,
                if thin { 8.0 } else { 2.0 },
            ),
        ];
        let rasters = [false, true].map(|inset| paint(SIZE, rect, f32::from(r), sigma, inset));
        for (name, xs, ys, limit) in regions {
            let expected = reference(rect, f64::from(r), f64::from(sigma), xs.clone(), ys.clone());
            for (inset, raster) in [false, true].into_iter().zip(&rasters) {
                let mut worst = (0.0, 0, 0);
                for (i, &mass) in expected.iter().enumerate() {
                    let x = usize::from(xs.start) + i % xs.len();
                    let y = usize::from(ys.start) + i / xs.len();
                    let alpha = f64::from(raster.data()[y * usize::from(SIZE.0) + x].a);
                    let error = (alpha - 255.0 * if inset { 1.0 - mass } else { mass }).abs();
                    if error > worst.0 {
                        worst = (error, x, y);
                    }
                }
                assert!(
                    worst.0 <= limit,
                    "{name}, rect {rect:?}, r={r}, blur={blur}, inset={inset}: maximum 2D alpha error {worst:?}, limit {limit}/255",
                );
            }
        }
    }
}

#[test]
fn sharp_corners_follow_the_2d_gaussian() {
    check_shape([64, 48, 128, 112], 0, false);
}

#[test]
fn rounded_card_follows_the_2d_gaussian() {
    check_shape([48, 40, 208, 136], 16, false);
}

#[test]
fn thin_bar_short_ends_and_corners_follow_the_2d_gaussian() {
    check_shape([40, 86, 216, 106], 4, true);
}
