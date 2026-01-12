//! Time interval types for historical data.

use serde::{Deserialize, Serialize};

/// Time interval for historical data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Interval {
    /// 1 minute bars
    Minute1,
    /// 5 minute bars
    Minute5,
    /// 15 minute bars
    Minute15,
    /// 30 minute bars
    Minute30,
    /// 1 hour bars
    Hour1,
    /// Daily bars
    Daily,
    /// Weekly bars
    Weekly,
    /// Monthly bars
    Monthly,
}

impl Interval {
    /// Convert to Yahoo Finance API string representation.
    pub fn as_yahoo_str(&self) -> &'static str {
        match self {
            Interval::Minute1 => "1m",
            Interval::Minute5 => "5m",
            Interval::Minute15 => "15m",
            Interval::Minute30 => "30m",
            Interval::Hour1 => "1h",
            Interval::Daily => "1d",
            Interval::Weekly => "1wk",
            Interval::Monthly => "1mo",
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Interval::Minute1 => "1 Minute",
            Interval::Minute5 => "5 Minutes",
            Interval::Minute15 => "15 Minutes",
            Interval::Minute30 => "30 Minutes",
            Interval::Hour1 => "1 Hour",
            Interval::Daily => "Daily",
            Interval::Weekly => "Weekly",
            Interval::Monthly => "Monthly",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_minute1_yahoo_str() {
        assert_eq!(Interval::Minute1.as_yahoo_str(), "1m");
    }

    #[test]
    fn test_interval_minute5_yahoo_str() {
        assert_eq!(Interval::Minute5.as_yahoo_str(), "5m");
    }

    #[test]
    fn test_interval_minute15_yahoo_str() {
        assert_eq!(Interval::Minute15.as_yahoo_str(), "15m");
    }

    #[test]
    fn test_interval_minute30_yahoo_str() {
        assert_eq!(Interval::Minute30.as_yahoo_str(), "30m");
    }

    #[test]
    fn test_interval_hour1_yahoo_str() {
        assert_eq!(Interval::Hour1.as_yahoo_str(), "1h");
    }

    #[test]
    fn test_interval_daily_yahoo_str() {
        assert_eq!(Interval::Daily.as_yahoo_str(), "1d");
    }

    #[test]
    fn test_interval_weekly_yahoo_str() {
        assert_eq!(Interval::Weekly.as_yahoo_str(), "1wk");
    }

    #[test]
    fn test_interval_monthly_yahoo_str() {
        assert_eq!(Interval::Monthly.as_yahoo_str(), "1mo");
    }

    #[test]
    fn test_interval_display_minute1() {
        assert_eq!(format!("{}", Interval::Minute1), "1 Minute");
    }

    #[test]
    fn test_interval_display_minute5() {
        assert_eq!(format!("{}", Interval::Minute5), "5 Minutes");
    }

    #[test]
    fn test_interval_display_minute15() {
        assert_eq!(format!("{}", Interval::Minute15), "15 Minutes");
    }

    #[test]
    fn test_interval_display_minute30() {
        assert_eq!(format!("{}", Interval::Minute30), "30 Minutes");
    }

    #[test]
    fn test_interval_display_hour1() {
        assert_eq!(format!("{}", Interval::Hour1), "1 Hour");
    }

    #[test]
    fn test_interval_display_daily() {
        assert_eq!(format!("{}", Interval::Daily), "Daily");
    }

    #[test]
    fn test_interval_display_weekly() {
        assert_eq!(format!("{}", Interval::Weekly), "Weekly");
    }

    #[test]
    fn test_interval_display_monthly() {
        assert_eq!(format!("{}", Interval::Monthly), "Monthly");
    }

    #[test]
    fn test_interval_clone() {
        let interval = Interval::Daily;
        let cloned = interval.clone();
        assert_eq!(interval, cloned);
    }

    #[test]
    fn test_interval_copy() {
        let interval = Interval::Weekly;
        let copied = interval;
        assert_eq!(interval, copied);
    }

    #[test]
    fn test_interval_equality() {
        assert_eq!(Interval::Daily, Interval::Daily);
        assert_ne!(Interval::Daily, Interval::Weekly);
    }

    #[test]
    fn test_interval_debug() {
        let debug_str = format!("{:?}", Interval::Monthly);
        assert_eq!(debug_str, "Monthly");
    }

    #[test]
    fn test_interval_serialize() {
        let interval = Interval::Daily;
        let json = serde_json::to_string(&interval).unwrap();
        assert_eq!(json, "\"Daily\"");
    }

    #[test]
    fn test_interval_deserialize() {
        let json = "\"Weekly\"";
        let interval: Interval = serde_json::from_str(json).unwrap();
        assert_eq!(interval, Interval::Weekly);
    }

    #[test]
    fn test_interval_serialize_all_variants() {
        let variants = vec![
            (Interval::Minute1, "\"Minute1\""),
            (Interval::Minute5, "\"Minute5\""),
            (Interval::Minute15, "\"Minute15\""),
            (Interval::Minute30, "\"Minute30\""),
            (Interval::Hour1, "\"Hour1\""),
            (Interval::Daily, "\"Daily\""),
            (Interval::Weekly, "\"Weekly\""),
            (Interval::Monthly, "\"Monthly\""),
        ];

        for (interval, expected) in variants {
            let json = serde_json::to_string(&interval).unwrap();
            assert_eq!(json, expected);
        }
    }

    #[test]
    fn test_interval_roundtrip_serialization() {
        let intervals = vec![
            Interval::Minute1,
            Interval::Minute5,
            Interval::Minute15,
            Interval::Minute30,
            Interval::Hour1,
            Interval::Daily,
            Interval::Weekly,
            Interval::Monthly,
        ];

        for interval in intervals {
            let json = serde_json::to_string(&interval).unwrap();
            let deserialized: Interval = serde_json::from_str(&json).unwrap();
            assert_eq!(interval, deserialized);
        }
    }
}
