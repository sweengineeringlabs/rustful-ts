//! Data source trait definition.

use crate::error::Result;
use crate::model::{Interval, Quote};

/// Trait for data sources that can fetch historical price data.
///
/// Implementations provide access to financial data from various providers.
pub trait DataSource: Send + Sync {
    /// Data source name.
    fn name(&self) -> &str;

    /// Fetch historical data synchronously.
    fn fetch_sync(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        interval: Interval,
    ) -> Result<Vec<Quote>>;
}

// Note: AsyncDataSource trait would be defined here if async-trait is added as a dependency
// For now, async support is handled directly in the implementations (data-core)

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DataError;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// Mock data source for testing
    struct MockDataSource {
        name: String,
        quotes: Vec<Quote>,
        call_count: Arc<AtomicUsize>,
        should_fail: bool,
        error_type: Option<DataError>,
    }

    impl MockDataSource {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                quotes: Vec::new(),
                call_count: Arc::new(AtomicUsize::new(0)),
                should_fail: false,
                error_type: None,
            }
        }

        fn with_quotes(mut self, quotes: Vec<Quote>) -> Self {
            self.quotes = quotes;
            self
        }

        fn with_failure(mut self, error: DataError) -> Self {
            self.should_fail = true;
            self.error_type = Some(error);
            self
        }

        fn get_call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl DataSource for MockDataSource {
        fn name(&self) -> &str {
            &self.name
        }

        fn fetch_sync(
            &self,
            _symbol: &str,
            _start_date: &str,
            _end_date: &str,
            _interval: Interval,
        ) -> Result<Vec<Quote>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);

            if self.should_fail {
                return Err(self.error_type.clone().unwrap_or(DataError::NoData));
            }

            Ok(self.quotes.clone())
        }
    }

    fn sample_quotes() -> Vec<Quote> {
        vec![
            Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 1000),
            Quote::new(1704153600, 103.0, 108.0, 102.0, 107.0, 106.5, 1200),
            Quote::new(1704240000, 107.0, 110.0, 105.0, 109.0, 108.5, 1100),
        ]
    }

    #[test]
    fn test_mock_data_source_name() {
        let source = MockDataSource::new("TestSource");
        assert_eq!(source.name(), "TestSource");
    }

    #[test]
    fn test_mock_data_source_empty_name() {
        let source = MockDataSource::new("");
        assert_eq!(source.name(), "");
    }

    #[test]
    fn test_fetch_sync_returns_quotes() {
        let quotes = sample_quotes();
        let source = MockDataSource::new("Test").with_quotes(quotes.clone());

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert!(result.is_ok());
        let fetched = result.unwrap();
        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].close, 103.0);
    }

    #[test]
    fn test_fetch_sync_empty_quotes() {
        let source = MockDataSource::new("Test");

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_fetch_sync_tracks_call_count() {
        let source = MockDataSource::new("Test").with_quotes(sample_quotes());

        assert_eq!(source.get_call_count(), 0);

        let _ = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);
        assert_eq!(source.get_call_count(), 1);

        let _ = source.fetch_sync("GOOGL", "2024-01-01", "2024-01-31", Interval::Weekly);
        assert_eq!(source.get_call_count(), 2);
    }

    #[test]
    fn test_fetch_sync_with_different_intervals() {
        let source = MockDataSource::new("Test").with_quotes(sample_quotes());

        let intervals = vec![
            Interval::Minute1,
            Interval::Minute5,
            Interval::Daily,
            Interval::Weekly,
            Interval::Monthly,
        ];

        for interval in intervals {
            let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", interval);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_fetch_sync_failure_no_data() {
        let source = MockDataSource::new("Test").with_failure(DataError::NoData);

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DataError::NoData));
    }

    #[test]
    fn test_fetch_sync_failure_request_failed() {
        let source = MockDataSource::new("Test")
            .with_failure(DataError::RequestFailed("Network error".to_string()));

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataError::RequestFailed(msg) => assert_eq!(msg, "Network error"),
            _ => panic!("Expected RequestFailed error"),
        }
    }

    #[test]
    fn test_fetch_sync_failure_invalid_date() {
        let source = MockDataSource::new("Test")
            .with_failure(DataError::InvalidDate("Bad date format".to_string()));

        let result = source.fetch_sync("AAPL", "invalid", "also-invalid", Interval::Daily);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataError::InvalidDate(msg) => assert_eq!(msg, "Bad date format"),
            _ => panic!("Expected InvalidDate error"),
        }
    }

    #[test]
    fn test_fetch_sync_failure_api_error() {
        let source = MockDataSource::new("Test").with_failure(DataError::ApiError {
            code: "429".to_string(),
            description: "Rate limit exceeded".to_string(),
        });

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert!(result.is_err());
        match result.unwrap_err() {
            DataError::ApiError { code, description } => {
                assert_eq!(code, "429");
                assert_eq!(description, "Rate limit exceeded");
            }
            _ => panic!("Expected ApiError"),
        }
    }

    #[test]
    fn test_data_source_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MockDataSource>();
    }

    #[test]
    fn test_data_source_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MockDataSource>();
    }

    #[test]
    fn test_data_source_trait_object() {
        let source: Box<dyn DataSource> =
            Box::new(MockDataSource::new("BoxedSource").with_quotes(sample_quotes()));

        assert_eq!(source.name(), "BoxedSource");

        let result = source.fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn test_multiple_data_sources() {
        let source1: Box<dyn DataSource> =
            Box::new(MockDataSource::new("Source1").with_quotes(sample_quotes()));
        let source2: Box<dyn DataSource> =
            Box::new(MockDataSource::new("Source2").with_quotes(vec![]));

        let sources: Vec<Box<dyn DataSource>> = vec![source1, source2];

        assert_eq!(sources[0].name(), "Source1");
        assert_eq!(sources[1].name(), "Source2");

        let result1 = sources[0].fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);
        let result2 = sources[1].fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily);

        assert_eq!(result1.unwrap().len(), 3);
        assert_eq!(result2.unwrap().len(), 0);
    }

    #[test]
    fn test_data_source_with_arc() {
        let source: Arc<dyn DataSource> =
            Arc::new(MockDataSource::new("ArcSource").with_quotes(sample_quotes()));

        let source_clone = Arc::clone(&source);

        assert_eq!(source.name(), "ArcSource");
        assert_eq!(source_clone.name(), "ArcSource");
    }

    /// Test that demonstrates typical usage pattern
    #[test]
    fn test_typical_usage_pattern() {
        // Create a data source
        let source = MockDataSource::new("Yahoo").with_quotes(sample_quotes());

        // Fetch data
        let quotes = source
            .fetch_sync("AAPL", "2024-01-01", "2024-01-31", Interval::Daily)
            .expect("Should fetch successfully");

        // Process data (using utility functions would be done elsewhere)
        assert!(!quotes.is_empty());
        assert_eq!(quotes[0].timestamp, 1704067200);
        assert_eq!(quotes[0].close, 103.0);
    }
}
