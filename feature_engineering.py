
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class UserFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.feature_columns_ = None

    @staticmethod
    def _mode(series):
        series = series.dropna()

        if series.empty:
            return np.nan

        return series.mode().iloc[0]

    @staticmethod
    def _clean_category_name(value):
        return (
            str(value)
            .lower()
            .replace(' ', '_')
            .replace('-', '_')
        )

    def _make_distribution_features(
        self,
        df,
        column,
        prefix
    ):
        shares = pd.crosstab(
            df['user_id'],
            df[column],
            normalize='index'
        )

        shares.columns = [
            f'share_{prefix}_{self._clean_category_name(value)}'
            for value in shares.columns
        ]

        max_share = shares.max(axis=1)
        max_share.name = f'{prefix}_max_share'

        probabilities = shares.replace(0, np.nan)

        entropy = -(
            probabilities * np.log(probabilities)
        ).sum(axis=1)

        entropy = entropy.fillna(0)
        entropy.name = f'{prefix}_entropy'

        return pd.concat(
            [shares, max_share, entropy],
            axis=1
        )

    def _create_features(self, X):

        df = X.copy()

        df['date'] = pd.to_datetime(df['date'])

        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = (
            df['dayofweek']
            .isin([5, 6])
            .astype(int)
        )

        grouped = df.groupby('user_id')

        features = grouped.agg(
            session_count=('session_id', 'nunique'),
            active_days=('date', 'nunique'),
            unique_website_categories=(
                'website_category',
                'nunique'
            ),
            weekend_share=('is_weekend', 'mean'),
            primary_device=('primary_device', self._mode),
            surf_depth=('surf_depth', self._mode),
            ads_activity=('ads_activity', self._mode),
            cloud_usage=('cloud_usage', self._mode),
            dominant_website_category=(
                'website_category',
                self._mode
            ),
            dominant_daytime=('daytime', self._mode),
            dominant_dayofweek=('dayofweek', self._mode),
            first_activity_date=('date', 'min'),
            last_activity_date=('date', 'max')
        )

        features['sessions_per_active_day'] = (
            features['session_count']
            / features['active_days']
        )

        features['activity_span_days'] = (
            features['last_activity_date']
            - features['first_activity_date']
        ).dt.days + 1

        features['active_days_ratio'] = (
            features['active_days']
            / features['activity_span_days']
        )

        features = features.drop(
            columns=[
                'first_activity_date',
                'last_activity_date'
            ]
        )

        website_features = self._make_distribution_features(
            df,
            'website_category',
            'website'
        )

        daytime_features = self._make_distribution_features(
            df,
            'daytime',
            'daytime'
        )

        dayofweek_features = self._make_distribution_features(
            df,
            'dayofweek',
            'dayofweek'
        )

        features = pd.concat(
            [
                features,
                website_features,
                daytime_features,
                dayofweek_features
            ],
            axis=1
        )

        categorical_features = [
            'primary_device',
            'surf_depth',
            'ads_activity',
            'cloud_usage',
            'dominant_website_category',
            'dominant_daytime',
            'dominant_dayofweek'
        ]

        for col in categorical_features:
            features[col] = features[col].apply(
                lambda x: str(x) if pd.notna(x) else np.nan
            )

        return features

    def fit(self, X, y=None):

        features = self._create_features(X)

        self.feature_columns_ = features.columns.tolist()

        return self

    def transform(self, X):

        features = self._create_features(X)

        features = features.reindex(
            columns=self.feature_columns_,
            fill_value=0
        )

        return features
