from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection strategies.
# Subclasses must implement the inspect method.
class DataInspectionStrategy(ABC):
    def __init__(self):
        self.text_col = None
        self.label_col = None
    
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be performed.

        Returns:
        None: This method prints the inspection results directly.
        """
        pass
    
# Concrete Strategy for Data Types Inspection
# --------------------------------------------
# This strategy inspects the data types of each column and counts non-null values.
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe columns.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the data types and non-null counts to the console.
        """
        print("\nData Types and Non-null Counts:")
        print(df.info())
        
# Concrete Strategy for Basic Info of the Data
# --------------------------------------------
# This strategy inspects the shape of the data, null values and duplicates values.
class BasicInfoInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the shape of the data, null values and duplicates values.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the shape of the data, null values and duplicates values.
        """
        print("Shape:", df.shape)
        print("Null values:\n", df.isnull().sum())
        print("Duplicate rows:", df.duplicated().sum())
        
# Concrete Strategy to Preview random samples
# --------------------------------------------
# This strategy inspects the data and prints few samples.
class RandomSamplesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints few samples.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints few random samples.
        """
        print("\nRandom Sample:\n", df.sample(5))
        
# Concrete Strategy to print the character count, word count and sentence count.
# --------------------------------------------
# This strategy inspects the data and printprints the character count, word count and sentence count.
class TextCountStatsInspectionStrategy(DataInspectionStrategy):
    def _auto_detect_text_column(self, df: pd.DataFrame) -> str:
        text_candidates = df.select_dtypes(include='object')
        avg_lengths = text_candidates.apply(lambda col: col.astype(str).apply(len).mean(), axis=0)
        return str(avg_lengths.idxmax())  # Returns column name as string

    def _auto_detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        non_numeric = df.select_dtypes(exclude='number').nunique()
        candidates = non_numeric[(non_numeric > 1) & (non_numeric < 20)]
        if not candidates.empty:
            idx = candidates.idxmax()
            return str(idx) if idx is not None else None
        else:
            return None
        
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the character count, word count and sentence count.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the character count, word count and sentence count.
        """
        self.text_col = self._auto_detect_text_column(df)
        self.label_col = self._auto_detect_label_column(df)

        if not self.text_col:
            raise ValueError("No text column detected.")
        if not self.label_col:
            raise ValueError("No label column detected.")
        
        nltk.download('punkt_tab')
        
        df['char_count'] = df[self.text_col].astype(str).apply(len)
        df['word_count'] = df[self.text_col].astype(str).apply(lambda x: len(x.split()))
        df['sentence_count'] = df[self.text_col].astype(str).apply(lambda x: len(sent_tokenize(x)))
        
        print("\nCharacter count stats:\n", df['char_count'].describe())
        print("\nWord count stats:\n", df['word_count'].describe())
        print("\nSentence count stats:\n", df['sentence_count'].describe())


# Concrete Strategy to print the stats of the character count, word count and sentence count.
# --------------------------------------------
# This strategy inspects the data and prints few samples.
class CountStatsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the stats of the character count, word count and sentence count.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Prints the stats of the character count, word count and sentence count.
        """
        print("\nMode (character count):", df['char_count'].mode()[0])
        print("\nMedian (character count):", df['char_count'].median())
        print("\nMean (character count):", df['char_count'].mean())
        
        print("\nMode (word count):", df['word_count'].mode()[0])
        print("\nMedian (word count):", df['word_count'].median())
        print("\nMean (word count):", df['word_count'].mean())
        
        print("\nMode (Sentence count):", df['sentence_count'].mode()[0])
        print("\nMedian (Sentence count):", df['sentence_count'].median())
        print("\nMean (Sentence count):", df['sentence_count'].mean())
        
# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategies.
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializes the DataInspector with a specific inspection strategy.

        Parameters:
        strategy (DataInspectionStrategy): The strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectionStrategy): The new strategy to be used for data inspection.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected.

        Returns:
        None: Executes the strategy's inspection method.
        """
        self._strategy.inspect(df)

# Example usage
if __name__ == "__main__":
    # Example usage of the DataInspector with different strategies.

    # Load the data
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize the Data Inspector with a specific strategy
    # inspector = DataInspector(DataTypesInspectionStrategy())
    # inspector.execute_inspection(df)

    # Change strategy to Summary Statistics and execute
    # inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    # inspector.execute_inspection(df)
    pass