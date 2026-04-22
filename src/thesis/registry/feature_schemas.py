from thesis.schemas.features import FeatureSchema

FEATURE_SCHEMAS = {
    "baseline": FeatureSchema(
        name="baseline",  # directly derived from one alert
        features=[
            "hour_of_day",
            "day_of_week",
            "is_internal_ip",
            "ip_freq",
            "host_freq",
            "short_freq",
        ],
    ),
    "dynamic": FeatureSchema(
        name="dynamic",
        features=[
            "short_count_1d",  # number of times the short (alert type) was seen in the last day
            "short_fp_rate_1d",  # false positive rate for the short in the last day
            "short_attack_rate_1d",  # attack rate for the short in the last day
            "host_count_1d",  # number of times the host was seen in the last day
            "host_fp_rate_1d",  # false positive rate for the host in the last day
            "host_attack_rate_1d",  # attack rate for the host in the last day
            "ip_count_1d",  # number of times the ip was seen in the last day
            "ip_fp_rate_1d",  # false positive rate for the ip in the last day
            "ip_attack_rate_1d",  # attack rate for the ip in the last day
            "short_host_count_1d",  # number of times the short-host combination was seen in the last day
            "short_host_fp_rate_1d",  # false positive rate for the short-host combination in the last day
            "short_host_attack_rate_1d",  # attack rate for the short-host combination in the last day
            "short_ip_count_1d",  # number of times the short-ip combination was seen in the last day
            "short_ip_fp_rate_1d",  # false positive rate for the short-ip combination in the last day
            "short_ip_attack_rate_1d",  # attack rate for the short-ip combination in the last day
            "seconds_since_short_seen",  # number of seconds since the short was last seen
            "seconds_since_host_seen",  # number of seconds since the host was last seen
            "seconds_since_ip_seen",  # number of seconds since the ip was last seen
            "seconds_since_short_host_seen",  # number of seconds since the short-host combination was last seen
        ],
    ),
}


def get_feature_schema(name: str) -> FeatureSchema:
    return FEATURE_SCHEMAS[name]
