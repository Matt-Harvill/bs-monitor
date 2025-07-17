"""Clear the cache for the BowelSoundDataset"""

from src.train.train import BowelSoundDataset


def clear_cache(data_dir: str) -> None:
    """Clear the cache for the BowelSoundDataset"""
    BowelSoundDataset.clear_all_cache(data_dir)


if __name__ == "__main__":
    clear_cache("data")
