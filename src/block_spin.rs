//! # Block-Spin Coarse-Graining Transformations
//!
//! Group nearby agents into blocks and compute block properties
//! (mean belief, aggregate constraint). This is the fundamental
//! RG operation: integrating out short-scale degrees of freedom.

use serde::{Deserialize, Serialize};

/// Represents a single agent in the fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique identifier
    pub id: usize,
    /// Position in space (1D for simplicity, generalizable)
    pub position: Vec<f64>,
    /// Agent's "spin" or belief state (continuous)
    pub belief: f64,
    /// Local constraint value (e.g., resource budget, bandwidth)
    pub constraint: f64,
}

/// A block of agents after coarse-graining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    /// Block identifier
    pub id: usize,
    /// Constituent agent IDs
    pub agents: Vec<usize>,
    /// Center of mass position
    pub center: Vec<f64>,
    /// Mean belief (magnetization analog)
    pub mean_belief: f64,
    /// Aggregate constraint
    pub aggregate_constraint: f64,
    /// Number of agents in block
    pub size: usize,
}

/// Configuration for a block transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockTransform {
    /// Current scale level
    pub scale: usize,
    /// Number of agents per block
    pub block_size: usize,
}

/// Strategy for grouping agents into blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockingStrategy {
    /// Sequential grouping: agents 0..b, b..2b, etc.
    Sequential,
    /// Spatial grouping based on nearest neighbors
    SpatialNearest,
    /// Majority rule: block spin = sign of mean belief
    MajorityRule,
}

impl BlockTransform {
    /// Create a new block transform at given scale with given block size.
    pub fn new(scale: usize, block_size: usize) -> Self {
        Self { scale, block_size }
    }

    /// Apply the block transformation to a set of agents.
    /// Returns the coarse-grained blocks.
    pub fn transform(&self, agents: &[Agent], strategy: BlockingStrategy) -> Vec<Block> {
        match strategy {
            BlockingStrategy::Sequential => self.sequential_blocking(agents),
            BlockingStrategy::SpatialNearest => self.spatial_blocking(agents),
            BlockingStrategy::MajorityRule => self.majority_rule_blocking(agents),
        }
    }

    fn sequential_blocking(&self, agents: &[Agent]) -> Vec<Block> {
        let b = self.block_size.max(1);
        let n = agents.len();
        let mut blocks = Vec::new();
        let mut block_id = 0;

        let mut i = 0;
        while i < n {
            let end = (i + b).min(n);
            let block_agents: Vec<usize> = (i..end).map(|j| agents[j].id).collect();
            let count = end - i;

            let mean_belief: f64 = (i..end).map(|j| agents[j].belief).sum::<f64>() / count as f64;
            let agg_constraint: f64 = (i..end).map(|j| agents[j].constraint).sum();

            let ndim = agents[i].position.len();
            let center: Vec<f64> = (0..ndim)
                .map(|d| (i..end).map(|j| agents[j].position[d]).sum::<f64>() / count as f64)
                .collect();

            blocks.push(Block {
                id: block_id,
                agents: block_agents,
                center,
                mean_belief,
                aggregate_constraint: agg_constraint,
                size: count,
            });
            block_id += 1;
            i += b;
        }
        blocks
    }

    fn spatial_blocking(&self, agents: &[Agent]) -> Vec<Block> {
        // For 1D: sort by position, then sequential block
        let mut indexed: Vec<usize> = (0..agents.len()).collect();
        indexed.sort_by(|&a, &b| {
            let pa = &agents[a].position;
            let pb = &agents[b].position;
            // Lexicographic comparison
            for i in 0..pa.len().min(pb.len()) {
                let cmp = pa[i].partial_cmp(&pb[i]).unwrap_or(std::cmp::Ordering::Equal);
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });

        let sorted_agents: Vec<Agent> = indexed.iter().map(|&i| agents[i].clone()).collect();
        let b = self.block_size.max(1);
        let n = sorted_agents.len();
        let mut blocks = Vec::new();
        let mut block_id = 0;

        let mut i = 0;
        while i < n {
            let end = (i + b).min(n);
            let count = end - i;
            let block_agents: Vec<usize> = (i..end).map(|j| sorted_agents[j].id).collect();

            let mean_belief: f64 = (i..end).map(|j| sorted_agents[j].belief).sum::<f64>() / count as f64;
            let agg_constraint: f64 = (i..end).map(|j| sorted_agents[j].constraint).sum();

            let ndim = sorted_agents[i].position.len();
            let center: Vec<f64> = (0..ndim)
                .map(|d| (i..end).map(|j| sorted_agents[j].position[d]).sum::<f64>() / count as f64)
                .collect();

            blocks.push(Block {
                id: block_id,
                agents: block_agents,
                center,
                mean_belief,
                aggregate_constraint: agg_constraint,
                size: count,
            });
            block_id += 1;
            i += b;
        }
        blocks
    }

    fn majority_rule_blocking(&self, agents: &[Agent]) -> Vec<Block> {
        let b = self.block_size.max(1);
        let n = agents.len();
        let mut blocks = Vec::new();
        let mut block_id = 0;

        let mut i = 0;
        while i < n {
            let end = (i + b).min(n);
            let count = end - i;
            let block_agents: Vec<usize> = (i..end).map(|j| agents[j].id).collect();

            let mean_belief: f64 = (i..end).map(|j| agents[j].belief).sum::<f64>() / count as f64;
            // Majority rule: block spin is ±1
            let block_spin = if mean_belief >= 0.0 { 1.0 } else { -1.0 };
            let agg_constraint: f64 = (i..end).map(|j| agents[j].constraint).sum();

            let ndim = agents[i].position.len();
            let center: Vec<f64> = (0..ndim)
                .map(|d| (i..end).map(|j| agents[j].position[d]).sum::<f64>() / count as f64)
                .collect();

            blocks.push(Block {
                id: block_id,
                agents: block_agents,
                center,
                mean_belief: block_spin,
                aggregate_constraint: agg_constraint,
                size: count,
            });
            block_id += 1;
            i += b;
        }
        blocks
    }
}

/// Convert blocks back into agents for the next RG step.
/// Each block becomes a single agent with the block's properties.
pub fn blocks_to_agents(blocks: &[Block]) -> Vec<Agent> {
    blocks
        .iter()
        .map(|b| Agent {
            id: b.id,
            position: b.center.clone(),
            belief: b.mean_belief,
            constraint: b.aggregate_constraint,
        })
        .collect()
}

/// Compute the total magnetization (mean belief) of a set of agents.
pub fn magnetization(agents: &[Agent]) -> f64 {
    if agents.is_empty() {
        return 0.0;
    }
    agents.iter().map(|a| a.belief).sum::<f64>() / agents.len() as f64
}

/// Compute the total constraint (sum of all agent constraints).
pub fn total_constraint(agents: &[Agent]) -> f64 {
    agents.iter().map(|a| a.constraint).sum()
}

/// Compute the energy of a 1D nearest-neighbor fleet (Ising-like).
/// E = -J Σ s_i s_{i+1} - h Σ s_i
pub fn ising_energy(agents: &[Agent], j_coupling: f64, h_field: f64) -> f64 {
    let n = agents.len();
    if n == 0 {
        return 0.0;
    }
    let mut energy = 0.0;
    for i in 0..n {
        energy -= h_field * agents[i].belief;
        if i + 1 < n {
            energy -= j_coupling * agents[i].belief * agents[i + 1].belief;
        }
    }
    energy
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_agents(n: usize, belief: f64, constraint: f64) -> Vec<Agent> {
        (0..n)
            .map(|i| Agent {
                id: i,
                position: vec![i as f64],
                belief,
                constraint,
            })
            .collect()
    }

    fn make_alternating_agents(n: usize) -> Vec<Agent> {
        (0..n)
            .map(|i| Agent {
                id: i,
                position: vec![i as f64],
                belief: if i % 2 == 0 { 1.0 } else { -1.0 },
                constraint: 1.0,
            })
            .collect()
    }

    #[test]
    fn test_block_transform_reduces_count() {
        let agents = make_uniform_agents(16, 1.0, 1.0);
        let bt = BlockTransform::new(1, 4);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        assert_eq!(blocks.len(), 4); // 16 / 4 = 4 blocks
    }

    #[test]
    fn test_block_transform_non_divisible() {
        let agents = make_uniform_agents(10, 1.0, 1.0);
        let bt = BlockTransform::new(1, 3);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        assert_eq!(blocks.len(), 4); // ceil(10/3) = 4 blocks
    }

    #[test]
    fn test_block_mean_belief_uniform() {
        let agents = make_uniform_agents(8, 0.7, 1.0);
        let bt = BlockTransform::new(1, 4);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        for b in &blocks {
            assert!((b.mean_belief - 0.7).abs() < 1e-10);
        }
    }

    #[test]
    fn test_block_aggregate_constraint() {
        let agents = make_uniform_agents(8, 1.0, 2.0);
        let bt = BlockTransform::new(1, 4);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        for b in &blocks {
            assert!((b.aggregate_constraint - 8.0).abs() < 1e-10); // 4 agents * 2.0
        }
    }

    #[test]
    fn test_block_center_position() {
        let agents = make_uniform_agents(4, 1.0, 1.0);
        let bt = BlockTransform::new(1, 2);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        assert!((blocks[0].center[0] - 0.5).abs() < 1e-10); // (0+1)/2
        assert!((blocks[1].center[0] - 2.5).abs() < 1e-10); // (2+3)/2
    }

    #[test]
    fn test_sequential_vs_spatial_sorted() {
        let agents = make_uniform_agents(8, 1.0, 1.0);
        let bt = BlockTransform::new(1, 2);
        let seq = bt.transform(&agents, BlockingStrategy::Sequential);
        let spat = bt.transform(&agents, BlockingStrategy::SpatialNearest);
        // For already-sorted input, sequential and spatial should give same result
        assert_eq!(seq.len(), spat.len());
        for (s, sp) in seq.iter().zip(spat.iter()) {
            assert!((s.mean_belief - sp.mean_belief).abs() < 1e-10);
        }
    }

    #[test]
    fn test_majority_rule_positive() {
        let agents = make_uniform_agents(6, 0.8, 1.0);
        let bt = BlockTransform::new(1, 3);
        let blocks = bt.transform(&agents, BlockingStrategy::MajorityRule);
        for b in &blocks {
            assert!((b.mean_belief - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_majority_rule_negative() {
        let agents = make_uniform_agents(6, -0.8, 1.0);
        let bt = BlockTransform::new(1, 3);
        let blocks = bt.transform(&agents, BlockingStrategy::MajorityRule);
        for b in &blocks {
            assert!((b.mean_belief - (-1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_magnetization_uniform() {
        let agents = make_uniform_agents(10, 1.0, 1.0);
        assert!((magnetization(&agents) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_magnetization_alternating() {
        let agents = make_alternating_agents(10);
        assert!(magnetization(&agents).abs() < 1e-10);
    }

    #[test]
    fn test_total_constraint_conservation() {
        let agents = make_uniform_agents(12, 1.0, 3.0);
        let total_before = total_constraint(&agents);

        let bt = BlockTransform::new(1, 4);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        let total_after: f64 = blocks.iter().map(|b| b.aggregate_constraint).sum();
        assert!((total_before - total_after).abs() < 1e-10);
    }

    #[test]
    fn test_blocks_to_agents_roundtrip() {
        let agents = make_uniform_agents(8, 1.0, 1.0);
        let bt = BlockTransform::new(1, 4);
        let blocks = bt.transform(&agents, BlockingStrategy::Sequential);
        let new_agents = blocks_to_agents(&blocks);
        assert_eq!(new_agents.len(), 2); // 8/4 = 2
        assert_eq!(new_agents[0].id, 0);
        assert_eq!(new_agents[1].id, 1);
    }

    #[test]
    fn test_iterative_coarse_graining() {
        let agents = make_uniform_agents(64, 1.0, 1.0);
        // Three levels of coarse-graining: 64 -> 16 -> 4 -> 1
        let bt1 = BlockTransform::new(1, 4);
        let blocks1 = bt1.transform(&agents, BlockingStrategy::Sequential);
        assert_eq!(blocks1.len(), 16);

        let agents2 = blocks_to_agents(&blocks1);
        let bt2 = BlockTransform::new(2, 4);
        let blocks2 = bt2.transform(&agents2, BlockingStrategy::Sequential);
        assert_eq!(blocks2.len(), 4);

        let agents3 = blocks_to_agents(&blocks2);
        let bt3 = BlockTransform::new(3, 4);
        let blocks3 = bt3.transform(&agents3, BlockingStrategy::Sequential);
        assert_eq!(blocks3.len(), 1);
    }
}
