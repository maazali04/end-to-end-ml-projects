import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class AgeImputer (BaseEstimator, TransformerMixin):

    def __init__(self):
        self.medians_ = {}
        self.global_median_ = None

    def _clean_title_string(self, title):
        if title == "Mme": return "Mrs"
        elif title in ["Mlle", "Ms"]: return "Miss"
        elif title in ["Countess", "Lady", "Sir", "Don", "Jonkheer"]: return "Royalty"
        elif title in ["Col", "Major", "Capt", "Dr", "Rev"]: return "Professional"
        return title

    def fit(self, x, y=None):
        x_df = pd.DataFrame(x).copy()
        titles = x_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        x_df['Title'] = titles.apply(self._clean_title_string)
        self.medians_ = x_df.groupby("Title")['Age'].median().to_dict()
        self.global_median_ = x_df["Age"].median()
        return self


    def transform(self, x):
        x_df = pd.DataFrame(x).copy()
        titles = x_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        x_df['Title'] = titles.apply(self._clean_title_string)

        x_df["Age"] = x_df.apply(
            lambda row: row['Age'] if pd.notnull(row['Age'])
            else self.medians_.get(row['Title'], self.global_median_), 
            axis=1
        )
        return x_df


class EmbarkedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mode_ = None
        
    def fit(self,x, y=None):
        x_df = pd.DataFrame(x).copy()
        self.mode_ = x_df["Embarked"].mode()[0]
        return self
        
    def transform(self,x):
        x_df = pd.DataFrame(x).copy()
        x_df["Embarked"] = x_df.apply(
            lambda row: row['Embarked'] if pd.notnull(row["Embarked"])
            else self.mode_,
            axis = 1
        )

        return x_df


class CabinImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_df = pd.DataFrame(x).copy()
        x_df['Cabin'] = x_df['Cabin'].notnull().astype(int)
        return x_df


class TicketImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.ticket_counts_ = {}

    def fit(self, x, y=None):
        x_df = pd.DataFrame(x).copy()
        self.ticket_counts_ = x_df["Ticket"].value_counts().to_dict()
        return self

    def transform(self, x):
        x_df = pd.DataFrame(x).copy()
        x_df['TicketGroupSize'] = x_df['Ticket'].map(self.ticket_counts_).fillna(1).astype(int)

        conditions = [
            (x_df['TicketGroupSize'] == 1),
            (x_df["TicketGroupSize"] >=2 ) & (x_df["TicketGroupSize"] <=4),
            (x_df["TicketGroupSize"] > 4)
        ]
        choices = ["Alone", "SmallGroup", "LargeGroup"]
        x_df["GroupType"] = np.select(conditions, choices, default="Alone")
        return x_df.drop(columns=["Name","PassengerId","Ticket"], errors="ignore")
    

class UniversalBackupImputer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.numeric_defaults_ = {}
        self.categorical_defaults_ = {}

    def fit(self, x, y=None):
        x_df = pd.DataFrame(x).copy()

        num_cols = x_df.select_dtypes(include=[np.number]).columns
        cat_cols = x_df.select_dtypes(exclude=[np.number]).columns

        for col in num_cols:
            self.numeric_defaults_[col] = x_df[col].median()

        for col in cat_cols:
            if not x_df[col].mode().empty:
                self.categorical_defaults_[col] = x_df[col].mode()[0]
            else:
                self.categorical_defaults_[col] = "Missing"

        return self

    def transform(self, x):
        x_df = pd.DataFrame(x).copy()
        for col, median_val in self.numeric_defaults_.items():
            if col in x_df.columns:
                x_df[col] = x_df[col].fillna(median_val)

        for col, mode_val in self.categorical_defaults_.items():
            if col in x_df.columns:
                x_df[col] = x_df[col].fillna(mode_val)

        return x_df


class LogFareTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self 

    def transform(self, x):
        x_df = pd.DataFrame(x).copy()
        
        if 'Fare' in x_df.columns:
            x_df['Log Fare'] = np.log1p(x_df['Fare'])
            
        return x_df

