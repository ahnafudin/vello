// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Serial chains of one or two independent-axis Gaussian stages, each with its own output crop, for renderers
//! that lower them into separately cropped intermediate layers. The queries exist with or without the `ahcut`
//! feature, which defines the stages; without it they find nothing.
use super::*;

/// A malformed chain is never interpreted as an unfiltered or truncated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaussianChainError {
    /// Unsupported node count, links, output, or primitive.
    Graph,
    /// Non-finite/negative region or invalid sigma.
    Parameters,
}

impl Filter {
    /// One or two independent-axis Gaussian stages, the second reading the first, with an output crop per stage
    /// and a final crop.
    #[cfg(feature = "ahcut")]
    pub fn gaussian_chain(
        stages: &[(FilterPrimitive, Rect)],
        final_region: Rect,
    ) -> Result<Self, GaussianChainError> {
        if !(1..=2).contains(&stages.len()) {
            return Err(GaussianChainError::Graph);
        }
        let mut graph = FilterGraph::new();
        for (index, (primitive, _)) in stages.iter().enumerate() {
            let input = if index == 0 {
                FilterInput::Source(FilterSource::SourceGraphic)
            } else {
                FilterInput::Result(FilterId((index - 1) as u16))
            };
            let id = graph.add(primitive.clone(), Some(FilterInputs::single(input)));
            graph.set_output(id);
        }
        graph.gaussian_regions = Some(stages.iter().map(|(_, region)| *region).collect());
        graph.gaussian_final_region = Some(final_region);
        graph
            .checked_gaussian_chain()?
            .ok_or(GaussianChainError::Graph)?;
        // A serial dependency consumes the sum of upstream and downstream halos.
        graph.filter_expansion = Rect::ZERO;
        graph.source_expansion = Rect::ZERO;
        for primitive in &graph.primitives {
            for (accumulated, halo) in [
                (&mut graph.filter_expansion, primitive.filter_expansion()),
                (&mut graph.source_expansion, primitive.source_expansion()),
            ] {
                *accumulated = Rect::new(
                    accumulated.x0 + halo.x0,
                    accumulated.y0 + halo.y0,
                    accumulated.x1 + halo.x1,
                    accumulated.y1 + halo.y1,
                );
            }
        }
        Ok(Self {
            graph: Arc::new(graph),
        })
    }

    /// The standard deviations and colour space when this filter is exactly one independent-axis Gaussian.
    pub fn single_gaussian_xy(&self) -> Option<([f32; 2], bool)> {
        match self.graph.primitives.as_slice() {
            #[cfg(feature = "ahcut")]
            [
                FilterPrimitive::GaussianBlurXY {
                    std_deviation,
                    linear_rgb,
                },
            ] => Some((*std_deviation, *linear_rgb)),
            _ => None,
        }
    }
}

impl FilterGraph {
    /// The final crop of a chain built by `Filter::gaussian_chain`, independent of the stage output crops.
    pub fn gaussian_final_region(&self) -> Option<Rect> {
        #[cfg(feature = "ahcut")]
        {
            self.gaussian_final_region
        }
        #[cfg(not(feature = "ahcut"))]
        {
            None
        }
    }

    /// The stages of a chain built by `Filter::gaussian_chain` after validating every link and parameter;
    /// `Ok(None)` for any other graph, which stays on the ordinary path.
    pub fn checked_gaussian_chain(
        &self,
    ) -> Result<Option<Vec<(FilterPrimitive, Rect)>>, GaussianChainError> {
        #[cfg(feature = "ahcut")]
        {
            self.checked_gaussian_stages()
        }
        #[cfg(not(feature = "ahcut"))]
        {
            Ok(None)
        }
    }

    #[cfg(feature = "ahcut")]
    fn checked_gaussian_stages(
        &self,
    ) -> Result<Option<Vec<(FilterPrimitive, Rect)>>, GaussianChainError> {
        let Some(regions) = &self.gaussian_regions else {
            return if self.gaussian_final_region.is_none() {
                Ok(None)
            } else {
                Err(GaussianChainError::Graph)
            };
        };
        let n = self.primitives.len();
        if !(1..=2).contains(&n)
            || self.inputs.len() != n
            || regions.len() != n
            || self.output != FilterId((n - 1) as u16)
        {
            return Err(GaussianChainError::Graph);
        }
        let final_region = self
            .gaussian_final_region
            .ok_or(GaussianChainError::Graph)?;
        fn valid(region: Rect) -> bool {
            [region.x0, region.y0, region.x1, region.y1]
                .iter()
                .all(|v| v.is_finite())
                && region.width() >= 0.0
                && region.height() >= 0.0
        }
        if !valid(final_region) {
            return Err(GaussianChainError::Parameters);
        }
        let mut stages = Vec::with_capacity(n);
        for (i, (primitive, region)) in self.primitives.iter().zip(regions).enumerate() {
            let Some(input) = &self.inputs[i] else {
                return Err(GaussianChainError::Graph);
            };
            let valid_input = matches!(
                (i, &input.primary),
                (0, FilterInput::Source(FilterSource::SourceGraphic))
                    | (1, FilterInput::Result(FilterId(0)))
            );
            if !valid_input || input.secondary.is_some() {
                return Err(GaussianChainError::Graph);
            }
            let FilterPrimitive::GaussianBlurXY { std_deviation, .. } = primitive else {
                return Err(GaussianChainError::Graph);
            };
            if !valid(*region)
                || crate::filter::gaussian_xy::transform_sigma(*std_deviation, &Affine::IDENTITY)
                    .is_err()
            {
                return Err(GaussianChainError::Parameters);
            }
            stages.push((primitive.clone(), *region));
        }
        Ok(Some(stages))
    }
}

#[cfg(all(test, feature = "ahcut"))]
mod tests {
    use super::*;
    #[test]
    fn retain_real_edges_crops_and_refuse_descriptor_mutation() {
        let f = Rect::new(80.0, 8.0, 168.0, 88.0);
        let small = Rect::new(120.0, 44.0, 124.0, 52.0);
        let a = FilterPrimitive::GaussianBlurXY {
            std_deviation: [1.0, 0.0],
            linear_rgb: false,
        };
        let b = FilterPrimitive::GaussianBlurXY {
            std_deviation: [0.0, 0.0],
            linear_rgb: false,
        };
        let filter = Filter::gaussian_chain(&[(a, f), (b, small)], f).unwrap();
        let stages = filter.graph.checked_gaussian_chain().unwrap().unwrap();
        assert_eq!(stages[1].1, small);
        assert_eq!(filter.single_gaussian_xy(), None);
        let mut bad = (*filter.graph).clone();
        bad.inputs[1] = Some(FilterInputs::single(FilterInput::Source(
            FilterSource::SourceGraphic,
        )));
        assert_eq!(bad.checked_gaussian_chain(), Err(GaussianChainError::Graph));
        bad = (*filter.graph).clone();
        bad.output = FilterId(0);
        assert_eq!(bad.checked_gaussian_chain(), Err(GaussianChainError::Graph));
    }
}
