//! Signal generator trait.

use crate::model::Signal;

/// Signal generator trait.
pub trait SignalGenerator: Send + Sync {
    /// Generate a signal based on current data.
    fn generate(&self, data: &[f64]) -> Signal;

    /// Generate signals for a series.
    fn generate_series(&self, data: &[f64]) -> Vec<Signal> {
        (1..=data.len())
            .map(|i| self.generate(&data[..i]))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Mock signal generator that returns Buy when last value > threshold
    struct MockThresholdGenerator {
        threshold: f64,
    }

    impl SignalGenerator for MockThresholdGenerator {
        fn generate(&self, data: &[f64]) -> Signal {
            match data.last() {
                Some(&value) if value > self.threshold => Signal::Buy,
                Some(&value) if value < self.threshold => Signal::Sell,
                _ => Signal::Hold,
            }
        }
    }

    /// Mock signal generator that always returns a fixed signal
    struct MockConstantGenerator {
        signal: Signal,
    }

    impl SignalGenerator for MockConstantGenerator {
        fn generate(&self, _data: &[f64]) -> Signal {
            self.signal
        }
    }

    /// Mock signal generator based on momentum (current vs previous)
    struct MockMomentumGenerator;

    impl SignalGenerator for MockMomentumGenerator {
        fn generate(&self, data: &[f64]) -> Signal {
            if data.len() < 2 {
                return Signal::Hold;
            }
            let current = data[data.len() - 1];
            let previous = data[data.len() - 2];
            if current > previous {
                Signal::Buy
            } else if current < previous {
                Signal::Sell
            } else {
                Signal::Hold
            }
        }
    }

    #[test]
    fn test_threshold_generator_buy() {
        let generator = MockThresholdGenerator { threshold: 100.0 };
        let data = vec![95.0, 98.0, 105.0];
        assert_eq!(generator.generate(&data), Signal::Buy);
    }

    #[test]
    fn test_threshold_generator_sell() {
        let generator = MockThresholdGenerator { threshold: 100.0 };
        let data = vec![105.0, 102.0, 95.0];
        assert_eq!(generator.generate(&data), Signal::Sell);
    }

    #[test]
    fn test_threshold_generator_hold() {
        let generator = MockThresholdGenerator { threshold: 100.0 };
        let data = vec![95.0, 98.0, 100.0];
        assert_eq!(generator.generate(&data), Signal::Hold);
    }

    #[test]
    fn test_constant_generator_buy() {
        let generator = MockConstantGenerator { signal: Signal::Buy };
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(generator.generate(&data), Signal::Buy);
    }

    #[test]
    fn test_constant_generator_sell() {
        let generator = MockConstantGenerator {
            signal: Signal::Sell,
        };
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(generator.generate(&data), Signal::Sell);
    }

    #[test]
    fn test_constant_generator_hold() {
        let generator = MockConstantGenerator {
            signal: Signal::Hold,
        };
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(generator.generate(&data), Signal::Hold);
    }

    #[test]
    fn test_momentum_generator_buy() {
        let generator = MockMomentumGenerator;
        let data = vec![100.0, 102.0];
        assert_eq!(generator.generate(&data), Signal::Buy);
    }

    #[test]
    fn test_momentum_generator_sell() {
        let generator = MockMomentumGenerator;
        let data = vec![102.0, 100.0];
        assert_eq!(generator.generate(&data), Signal::Sell);
    }

    #[test]
    fn test_momentum_generator_hold() {
        let generator = MockMomentumGenerator;
        let data = vec![100.0, 100.0];
        assert_eq!(generator.generate(&data), Signal::Hold);
    }

    #[test]
    fn test_momentum_generator_insufficient_data() {
        let generator = MockMomentumGenerator;
        let data = vec![100.0];
        assert_eq!(generator.generate(&data), Signal::Hold);
    }

    #[test]
    fn test_generate_series_basic() {
        let generator = MockConstantGenerator { signal: Signal::Buy };
        let data = vec![1.0, 2.0, 3.0];
        let signals = generator.generate_series(&data);
        assert_eq!(signals.len(), 3);
        assert!(signals.iter().all(|s| *s == Signal::Buy));
    }

    #[test]
    fn test_generate_series_momentum() {
        let generator = MockMomentumGenerator;
        let data = vec![100.0, 102.0, 101.0, 103.0];
        let signals = generator.generate_series(&data);

        assert_eq!(signals.len(), 4);
        assert_eq!(signals[0], Signal::Hold); // Only one data point
        assert_eq!(signals[1], Signal::Buy); // 102 > 100
        assert_eq!(signals[2], Signal::Sell); // 101 < 102
        assert_eq!(signals[3], Signal::Buy); // 103 > 101
    }

    #[test]
    fn test_generate_series_empty_data() {
        let generator = MockConstantGenerator { signal: Signal::Buy };
        let data: Vec<f64> = vec![];
        let signals = generator.generate_series(&data);
        assert!(signals.is_empty());
    }

    #[test]
    fn test_generate_series_single_element() {
        let generator = MockMomentumGenerator;
        let data = vec![100.0];
        let signals = generator.generate_series(&data);
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0], Signal::Hold);
    }

    #[test]
    fn test_signal_generator_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MockThresholdGenerator>();
        assert_send::<MockConstantGenerator>();
        assert_send::<MockMomentumGenerator>();
    }

    #[test]
    fn test_signal_generator_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MockThresholdGenerator>();
        assert_sync::<MockConstantGenerator>();
        assert_sync::<MockMomentumGenerator>();
    }

    #[test]
    fn test_signal_generator_in_arc() {
        let generator: Arc<dyn SignalGenerator> = Arc::new(MockConstantGenerator { signal: Signal::Buy });
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(generator.generate(&data), Signal::Buy);
    }

    #[test]
    fn test_signal_generator_trait_object() {
        let generators: Vec<Box<dyn SignalGenerator>> = vec![
            Box::new(MockConstantGenerator { signal: Signal::Buy }),
            Box::new(MockConstantGenerator { signal: Signal::Sell }),
            Box::new(MockConstantGenerator { signal: Signal::Hold }),
        ];

        let data = vec![1.0, 2.0];
        let signals: Vec<Signal> = generators.iter().map(|g| g.generate(&data)).collect();

        assert_eq!(signals, vec![Signal::Buy, Signal::Sell, Signal::Hold]);
    }

    #[test]
    fn test_generate_series_incremental_view() {
        let generator = MockThresholdGenerator { threshold: 50.0 };
        let data = vec![40.0, 60.0, 45.0, 55.0];
        let signals = generator.generate_series(&data);

        // Each signal is generated from data[..i+1]:
        // data[..1] = [40.0] -> last is 40 < 50 -> Sell
        // data[..2] = [40.0, 60.0] -> last is 60 > 50 -> Buy
        // data[..3] = [40.0, 60.0, 45.0] -> last is 45 < 50 -> Sell
        // data[..4] = [40.0, 60.0, 45.0, 55.0] -> last is 55 > 50 -> Buy
        assert_eq!(signals, vec![Signal::Sell, Signal::Buy, Signal::Sell, Signal::Buy]);
    }

    #[test]
    fn test_generator_with_large_dataset() {
        let generator = MockMomentumGenerator;
        let data: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 100.0 + 100.0).collect();
        let signals = generator.generate_series(&data);
        assert_eq!(signals.len(), 1000);
    }
}
