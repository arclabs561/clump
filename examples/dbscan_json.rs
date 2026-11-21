//! Read one JSON array of points from stdin and write their DBSCAN labels.
use clump::Dbscan;
use std::{env, error::Error, io};

fn cluster(
    input: impl io::Read,
    radius: f32,
    min_points: usize,
) -> Result<Vec<Option<usize>>, Box<dyn Error>> {
    if !radius.is_finite() || radius <= 0.0 || min_points == 0 {
        return Err("expected a finite positive radius and at least one point".into());
    }
    let points: Vec<Vec<f32>> = serde_json::from_reader(input)?;
    Ok(Dbscan::new(radius, min_points).fit_predict_with_noise(&points)?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let [radius, min_points] = args.as_slice() else {
        return Err("usage: dbscan_json RADIUS MIN_POINTS < points.json".into());
    };
    let labels = cluster(io::stdin().lock(), radius.parse()?, min_points.parse()?)?;
    println!("{}", serde_json::to_string(&labels)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smaller_group_becomes_noise_when_minimum_increases() {
        let input = b"[[0,0],[10,10],[0.1,0],[10.1,10],[0,0.1],[30,30]]";
        let labels = cluster(&input[..], 0.25, 2).unwrap();
        assert_eq!(labels, [Some(0), Some(1), Some(0), Some(1), Some(0), None]);
        let labels = cluster(&input[..], 0.25, 3).unwrap();
        assert_eq!(labels, [Some(0), None, Some(0), None, Some(0), None]);
    }

    #[test]
    fn rejects_bad_dimensions_and_parameters() {
        assert!(cluster(&b"[[0],[1,2]]"[..], 1.0, 2).is_err());
        for radius in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert!(cluster(&b"[[0]]"[..], radius, 1).is_err());
        }
        assert!(cluster(&b"[[0]]"[..], 1.0, 0).is_err());
    }
}
