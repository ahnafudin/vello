// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Independent-axis Gaussian blur with a colour space: mapping user-space standard deviations through a
//! transform, and the prepared blur that renderers execute.

use crate::filter::gaussian_blur::GaussianBlur;
use crate::filter_effects::{EdgeMode, Filter};
use crate::kurbo::Affine;
#[cfg(not(feature = "std"))]
use peniko::kurbo::common::FloatFuncs as _;

/// Why a covariance cannot be mapped exactly onto two axis passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianXYError {
    /// A standard deviation is negative or not finite.
    InvalidSigma,
    /// The transform rotates or skews the covariance off the axes, so two axis passes cannot represent it.
    UnsupportedTransform,
}

impl core::fmt::Display for GaussianXYError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Maps a diagonal covariance through `transform` when it stays diagonal (scales, translations and quarter turns);
/// the axes are never averaged.
pub fn transform_sigma(sigma: [f32; 2], transform: &Affine) -> Result<[f32; 2], GaussianXYError> {
    if sigma.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(GaussianXYError::InvalidSigma);
    }
    let [a, b, c, d, e, f] = transform.as_coeffs();
    if ![a, b, c, d, e, f].iter().all(|v| v.is_finite()) {
        return Err(GaussianXYError::UnsupportedTransform);
    }
    // Rotations of a quarter turn come with rounding error in the zero coefficients.
    let eps = 1e-6;
    let result = if b.abs() < eps && c.abs() < eps && a != 0.0 && d != 0.0 {
        [f64::from(sigma[0]) * a.abs(), f64::from(sigma[1]) * d.abs()]
    } else if a.abs() < eps && d.abs() < eps && b != 0.0 && c != 0.0 {
        [f64::from(sigma[1]) * c.abs(), f64::from(sigma[0]) * b.abs()]
    } else {
        return Err(GaussianXYError::UnsupportedTransform);
    };
    if result.iter().any(|v| !v.is_finite()) {
        return Err(GaussianXYError::InvalidSigma);
    }
    Ok([result[0] as f32, result[1] as f32])
}

/// The axis standard deviations of the transformed covariance. Where [`transform_sigma`] succeeds they are the
/// same; under a rotation or skew the correlation term is dropped, giving the closest axis-aligned Gaussian (as
/// the uniform blur approximates a non-uniform scale). A negative or non-finite input blurs nothing on its axis.
pub fn transform_sigma_axis_aligned(sigma: [f32; 2], transform: &Affine) -> [f32; 2] {
    if let Ok(exact) = transform_sigma(sigma, transform) {
        return exact;
    }
    let s = sigma.map(|v| {
        if v.is_finite() && v > 0.0 {
            f64::from(v)
        } else {
            0.0
        }
    });
    let [a, b, c, d, _, _] = transform.as_coeffs();
    let sx = ((a * s[0]) * (a * s[0]) + (c * s[1]) * (c * s[1])).sqrt();
    let sy = ((b * s[0]) * (b * s[0]) + (d * s[1]) * (d * s[1])).sqrt();
    [sx, sy].map(|v| if v.is_finite() { v as f32 } else { 0.0 })
}

/// A prepared independent-axis Gaussian blur in device space.
#[derive(Debug)]
pub struct PreparedGaussianXY {
    /// The blur, with its decimation plan and kernel per axis.
    pub blur: GaussianBlur,
    /// Blur in linear-light premultiplied colour instead of encoded sRGB.
    pub linear_rgb: bool,
}

impl PreparedGaussianXY {
    /// Prepares the blur for user-space standard deviations drawn under `transform`.
    pub fn new(sigma: [f32; 2], linear_rgb: bool, transform: &Affine) -> Self {
        let device = transform_sigma_axis_aligned(sigma, transform);
        Self {
            blur: GaussianBlur::new(device[0], device[1], EdgeMode::None),
            linear_rgb,
        }
    }

    /// The prepared blur when `filter` is a single independent-axis Gaussian, the one filter kind a renderer must
    /// take from here rather than from [`PreparedFilter`](crate::filter::PreparedFilter). Always `None` without
    /// the `ahcut` feature, which defines that kind.
    pub fn from_filter(filter: &Filter, transform: &Affine) -> Option<Self> {
        let (sigma, linear_rgb) = filter.single_gaussian_xy()?;
        Some(Self::new(sigma, linear_rgb, transform))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_scale_quarter_turn_and_rotation() {
        assert_eq!(
            transform_sigma([2.0, 3.0], &Affine::scale_non_uniform(2.0, 4.0)),
            Ok([4.0, 12.0])
        );
        let quarter =
            transform_sigma([2.0, 0.0], &Affine::rotate(core::f64::consts::FRAC_PI_2)).unwrap();
        assert!(quarter[0].abs() < 1e-5 && (quarter[1] - 2.0).abs() < 1e-5);
        assert_eq!(
            transform_sigma([2.0, 0.0], &Affine::rotate(0.3)),
            Err(GaussianXYError::UnsupportedTransform)
        );
        assert_eq!(
            transform_sigma([-1.0, 0.0], &Affine::IDENTITY),
            Err(GaussianXYError::InvalidSigma)
        );
    }

    #[test]
    fn axis_aligned_fallback_keeps_exact_cases_and_bounds_rotations() {
        let scale = Affine::scale_non_uniform(2.0, 4.0);
        assert_eq!(
            transform_sigma_axis_aligned([2.0, 3.0], &scale),
            [4.0, 12.0]
        );
        let eighth =
            transform_sigma_axis_aligned([4.0, 0.0], &Affine::rotate(core::f64::consts::FRAC_PI_4));
        let expected = 4.0 / 2.0_f32.sqrt();
        assert!((eighth[0] - expected).abs() < 1e-4 && (eighth[1] - expected).abs() < 1e-4);
        assert_eq!(
            transform_sigma_axis_aligned([f32::NAN, -1.0], &Affine::rotate(0.3)),
            [0.0, 0.0]
        );
    }
}
