// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Alpha across a straight edge of the blurred rectangle used for CSS box shadows.
#![cfg(feature = "ahcut")]

use core::f64::consts::{SQRT_2, TAU};
use vello_cpu::color::palette::css;
use vello_cpu::kurbo::{Rect, Shape};
use vello_cpu::{Pixmap, RenderContext, Resources};

const EDGE: i32 = 96;
const SIZE: u16 = 128;

fn paint(sigma: f32, inset: bool) -> Pixmap {
    let mut ctx = RenderContext::new(SIZE, SIZE);
    ctx.set_paint(css::BLACK);
    // A 64 px square offset by 24 px from (8, 8), leaving its far edges exposed.
    let shadow = Rect::new(32.0, 32.0, f64::from(EDGE), f64::from(EDGE));
    let area = Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)).to_path(0.1);
    ctx.fill_blurred_rounded_rect_in(&area, &shadow, 0.0, sigma, inset);
    let mut target = Pixmap::new(SIZE, SIZE);
    ctx.flush();
    ctx.render(&mut target, &mut Resources::new());
    target
}

// An antiderivative of the Gaussian CDF, evaluated at the two pixel boundaries.
fn integrated_cdf(d: f64, sigma: f64) -> f64 {
    let z = d / sigma;
    d * (1.0 + libm::erf(z / SQRT_2)) * 0.5 + sigma * (-0.5 * z * z).exp() / TAU.sqrt()
}

fn pixel_alpha(offset: i32, sigma: f64) -> f64 {
    let d = f64::from(-offset);
    integrated_cdf(d, sigma) - integrated_cdf(d - 1.0, sigma)
}

fn check_profile(raster: &Pixmap, radius: f64, inset: bool, vertical: bool) {
    let sigma = radius / 2.0;
    let offsets = -12..12;
    let samples: Vec<_> = offsets
        .clone()
        .map(|offset| {
            let at = (EDGE + offset) as usize;
            let (x, y) = if vertical { (64, at) } else { (at, 64) };
            let alpha = f64::from(raster.data()[y * usize::from(SIZE) + x].a) / 255.0;
            if inset { 1.0 - alpha } else { alpha }
        })
        .collect();
    if radius == 0.0 {
        for (offset, &alpha) in offsets.zip(&samples) {
            assert_eq!(alpha, if offset < 0 { 1.0 } else { 0.0 });
        }
        return;
    }
    let squared_error = |trial| {
        offsets
            .clone()
            .zip(&samples)
            .map(|(offset, &alpha)| (alpha - pixel_alpha(offset, trial)).powi(2))
            .sum::<f64>()
    };
    // Fit sigma over 0.5 to 5 px, at 0.01 px resolution.
    let fit = (50..=500)
        .map(|n| f64::from(n) / 100.0)
        .map(|trial| (trial, squared_error(trial)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0;
    let residual = offsets
        .zip(&samples)
        .map(|(offset, &alpha)| (alpha - pixel_alpha(offset, sigma)).abs())
        .fold(0.0, f64::max);
    assert!(
        (fit - sigma).abs() <= 0.1_f64.max(0.04 * sigma) && residual <= 2.0 / 255.0,
        "blur {radius}, inset {inset}, vertical edge scan {vertical}: fitted sigma {fit}, target {sigma}, maximum alpha error {residual}"
    );
}

fn check_css_radius(radius: f64) {
    for inset in [false, true] {
        let raster = paint((radius / 2.0) as f32, inset);
        for vertical in [false, true] {
            check_profile(&raster, radius, inset, vertical);
        }
    }
}

#[test]
fn zero_radius_has_a_sharp_edge() {
    check_css_radius(0.0);
}

#[test]
fn five_pixel_blur_matches_the_css_shadow_profile() {
    check_css_radius(5.0);
}

#[test]
fn six_pixel_blur_matches_the_css_shadow_profile() {
    check_css_radius(6.0);
}
