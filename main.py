#!/usr/bin/env python3
"""
Wedding playlist reviewer with ML-assisted progressive filtering

Usage:
  python main.py [dataset.csv]

What it does:
- Genre category selection to exclude unwanted categories
- Review songs in batches of 15 with 1-4 ratings
- ML model learns preferences and filters to top 50% after each batch
- Goal: Find 60 songs rated 4 (love it)
- Saves progress so you can resume later
- Exports songs by rating to CSV files
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


CHOICES_HELP = """
Rating System:
  1        Don't like
  2        Neutral / Maybe
  3        Like
  4        Love it! (highest rating)
  
Other:
  q        quit (progress is saved)
  ?        show this help
"""


@dataclass
class SongRow:
    category: str
    song: str
    artist: str
    dance_speed: str
    # Optional ML features (will be None if not in CSV)
    bpm: Optional[float] = None
    energy: Optional[float] = None
    valence: Optional[float] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    # Additional Spotify features
    danceability: Optional[float] = None
    popularity: Optional[int] = None
    duration_ms: Optional[int] = None
    loudness: Optional[float] = None
    acousticness: Optional[float] = None

    @staticmethod
    def from_dict(d: Dict[str, str]) -> "SongRow":
        def safe_float(key: str) -> Optional[float]:
            val = d.get(key, "").strip()
            if not val:
                return None
            try:
                return float(val)
            except ValueError:
                return None

        def safe_int(key: str) -> Optional[int]:
            val = d.get(key, "").strip()
            if not val:
                return None
            try:
                return int(val)
            except ValueError:
                return None

        # Spotify dataset format
        track_name = d.get("track_name", "").strip()
        artists = d.get("artists", "").strip()
        track_genre = d.get("track_genre", "").strip()
        # Derive dance_speed from tempo or danceability
        tempo = safe_float("tempo")
        danceability_val = safe_float("danceability")
        if tempo:
            # Fast if tempo > 120 BPM, Slow otherwise
            dance_speed = "Fast" if tempo > 120 else "Slow"
        elif danceability_val:
            # Fast if danceability > 0.6, Slow otherwise
            dance_speed = "Fast" if danceability_val > 0.6 else "Slow"
        else:
            dance_speed = "Unknown"
        
        category = track_genre or "Unknown"
        song = track_name
        artist = artists
        genre = track_genre
        bpm = tempo
        energy = safe_float("energy")
        valence = safe_float("valence")
        year = None  # Not in Spotify dataset
        danceability = danceability_val
        popularity = safe_int("popularity")
        duration_ms = safe_int("duration_ms")
        loudness = safe_float("loudness")
        acousticness = safe_float("acousticness")

        return SongRow(
            category=category,
            song=song,
            artist=artist,
            dance_speed=dance_speed,
            bpm=bpm,
            energy=energy,
            valence=valence,
            year=year,
            genre=genre,
            danceability=danceability,
            popularity=popularity,
            duration_ms=duration_ms,
            loudness=loudness,
            acousticness=acousticness,
        )

    def to_dict(self) -> Dict[str, str]:
        # Return in old format for compatibility
        result = {
            "Category": self.category,
            "Song": self.song,
            "Artist": self.artist,
            "DanceSpeed": self.dance_speed,
        }
        if self.bpm is not None:
            result["BPM"] = str(self.bpm)
        if self.energy is not None:
            result["Energy"] = str(self.energy)
        if self.valence is not None:
            result["Valence"] = str(self.valence)
        if self.year is not None:
            result["Year"] = str(self.year)
        if self.genre is not None:
            result["Genre"] = self.genre
        if self.danceability is not None:
            result["Danceability"] = str(self.danceability)
        if self.popularity is not None:
            result["Popularity"] = str(self.popularity)
        return result


def get_genre_category(genre: str) -> str:
    """Map individual genres to broader categories."""
    genre_lower = genre.lower()
    
    # Pop
    if genre_lower in {'pop', 'pop-film', 'power-pop', 'synth-pop', 'indie-pop'}:
        return 'Pop'
    
    # Rock
    if genre_lower in {'rock', 'alt-rock', 'alternative', 'hard-rock', 'rock-n-roll', 'rockabilly', 'psych-rock'}:
        return 'Rock'
    
    # Metal
    if genre_lower in {'metal', 'black-metal', 'death-metal', 'heavy-metal', 'metalcore', 'grindcore'}:
        return 'Metal'
    
    # Punk
    if genre_lower in {'punk', 'punk-rock', 'emo'}:
        return 'Punk'
    
    # Hip-Hop & R&B
    if genre_lower in {'hip-hop', 'r-n-b', 'rap'}:
        return 'Hip-Hop & R&B'
    
    # Electronic & Dance
    if genre_lower in {'edm', 'electronic', 'electro', 'dance', 'house', 'deep-house', 'progressive-house', 
                       'techno', 'minimal-techno', 'detroit-techno', 'chicago-house', 'trance', 
                       'dubstep', 'drum-and-bass', 'breakbeat', 'garage', 'hardstyle', 'hardcore', 
                       'idm', 'club'}:
        return 'Electronic & Dance'
    
    # Indie
    if genre_lower in {'indie', 'british'}:
        return 'Indie'
    
    # Country
    if genre_lower in {'country', 'honky-tonk', 'bluegrass'}:
        return 'Country'
    
    # Jazz
    if genre_lower == 'jazz':
        return 'Jazz'
    
    # Blues
    if genre_lower == 'blues':
        return 'Blues'
    
    # Folk
    if genre_lower in {'folk', 'singer-songwriter', 'songwriter'}:
        return 'Folk'
    
    # Classical
    if genre_lower in {'classical', 'opera', 'piano'}:
        return 'Classical'
    
    # Latin
    if genre_lower in {'latin', 'latino', 'salsa', 'samba', 'reggaeton', 'brazil', 'forro', 
                       'sertanejo', 'pagode', 'tango', 'mpb'}:
        return 'Latin'
    
    # International/World
    if genre_lower in {'world-music', 'indian', 'iranian', 'turkish', 'swedish', 'french', 'german', 
                       'spanish', 'malay'}:
        return 'International'
    
    # K-Pop & J-Pop
    if genre_lower in {'k-pop', 'j-pop', 'j-rock', 'j-dance', 'j-idol', 'anime', 'cantopop', 'mandopop'}:
        return 'K-Pop & J-Pop'
    
    # Reggae
    if genre_lower in {'reggae', 'dancehall', 'dub'}:
        return 'Reggae'
    
    # Funk & Soul
    if genre_lower in {'funk', 'soul', 'groove'}:
        return 'Funk & Soul'
    
    # Gospel
    if genre_lower == 'gospel':
        return 'Gospel'
    
    # Disco
    if genre_lower == 'disco':
        return 'Disco'
    
    # Ambient & Chill
    if genre_lower in {'ambient', 'chill', 'sleep', 'study', 'new-age'}:
        return 'Ambient & Chill'
    
    # Acoustic
    if genre_lower in {'acoustic', 'guitar'}:
        return 'Acoustic'
    
    # Mood/Theme
    if genre_lower in {'party', 'happy', 'sad', 'romance'}:
        return 'Mood & Theme'
    
    # Children
    if genre_lower in {'children', 'kids', 'disney', 'show-tunes'}:
        return 'Children'
    
    # Comedy
    if genre_lower == 'comedy':
        return 'Comedy'
    
    # Industrial & Goth
    if genre_lower in {'industrial', 'goth'}:
        return 'Industrial & Goth'
    
    # Grunge
    if genre_lower == 'grunge':
        return 'Grunge'
    
    # Trip-Hop
    if genre_lower == 'trip-hop':
        return 'Trip-Hop'
    
    # Afrobeat
    if genre_lower == 'afrobeat':
        return 'Afrobeat'
    
    # Default: Other
    return 'Other'


def model_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".model.pkl")


def extract_features(rows: List[SongRow]) -> np.ndarray:
    """Extract features from song rows for ML model."""
    if not ML_AVAILABLE:
        raise RuntimeError("scikit-learn not available. Install with: pip install scikit-learn numpy")

    features = []
    for row in rows:
        feat = []
        # Category/Genre (hash encoding)
        feat.append(hash(row.category) % 1000)
        # DanceSpeed (binary: Fast=1, Slow=0)
        feat.append(1.0 if row.dance_speed.lower() == "fast" else 0.0)
        # Artist (hash encoding)
        feat.append(hash(row.artist) % 1000)
        # Core audio features (use Spotify data if available, otherwise defaults)
        feat.append(row.bpm if row.bpm is not None else 0.0)
        feat.append(row.energy if row.energy is not None else 0.0)
        feat.append(row.valence if row.valence is not None else 0.0)
        feat.append(row.danceability if row.danceability is not None else 0.0)
        # Additional Spotify features (if available)
        feat.append(float(row.popularity) if row.popularity is not None else 0.0)
        feat.append(row.loudness if row.loudness is not None else 0.0)
        feat.append(row.acousticness if row.acousticness is not None else 0.0)
        # Year (if available)
        feat.append(float(row.year) if row.year is not None else 0.0)
        # Genre (hash encoding if available and different from category)
        if row.genre and row.genre != row.category:
            feat.append(hash(row.genre) % 1000)
        else:
            feat.append(0.0)
        features.append(feat)
    return np.array(features)


def train_model(rows: List[SongRow], decisions: Dict[str, str], min_samples: int = 15) -> Optional[Tuple]:
    """Train ML model on ratings (1-4). Returns (model, scaler, label_encoders) or None if insufficient data."""
    if not ML_AVAILABLE:
        return None

    # Collect training data
    X_train = []
    y_train = []
    for idx, row in enumerate(rows):
        rating_str = decisions.get(str(idx))
        if rating_str and rating_str in {"1", "2", "3", "4"}:
            X_train.append(row)
            y_train.append(int(rating_str))

    if len(y_train) < min_samples:
        return None

    # Extract features
    X_features = extract_features(X_train)
    y_array = np.array(y_train)

    # Train model (classification with 4 classes: 1, 2, 3, 4)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_features, y_array)

    return (model, None, None)  # No scaler/encoders needed with hash encoding


def predict_song(model_data: Optional[Tuple], row: SongRow) -> Optional[Tuple[int, float]]:
    """Predict rating (1-4) for a song. Returns (predicted_rating, confidence) or None."""
    if not ML_AVAILABLE or model_data is None:
        return None

    model, _, _ = model_data
    features = extract_features([row])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(probabilities))

    return (int(prediction), confidence)


def predict_scores(rows: List[SongRow], indices: List[int], model_data: Optional[Tuple]) -> Dict[int, float]:
    """Predict expected rating (weighted average). Returns dict mapping index to expected_rating."""
    if not ML_AVAILABLE or model_data is None:
        return {}
    
    model, _, _ = model_data
    song_rows = [rows[idx] for idx in indices]
    features = extract_features(song_rows)
    
    # Get probability distribution over all ratings
    probabilities = model.predict_proba(features)
    class_names = model.classes_
    
    # Calculate expected rating: sum(rating * probability) for each song
    scores = {}
    for i, idx in enumerate(indices):
        expected_rating = 0.0
        for j, cls in enumerate(class_names):
            rating = int(cls)
            prob = float(probabilities[i][j])
            expected_rating += rating * prob
        scores[idx] = expected_rating
    
    return scores


def filter_to_top_percentile(
    rows: List[SongRow],
    active_subset: List[int],
    decisions: Dict[str, str],
    model_data: Optional[Tuple],
    percentile: float = 0.5,
    min_size: int = 200
) -> List[int]:
    """Filter active subset to top percentile based on ML predictions. Returns new active subset."""
    # Get unreviewed songs from active subset
    unreviewed = [idx for idx in active_subset if str(idx) not in decisions]
    
    if not unreviewed or model_data is None:
        # If no model or no unreviewed songs, return current subset
        return active_subset
    
    # Predict scores for unreviewed songs
    scores = predict_scores(rows, unreviewed, model_data)
    
    if not scores:
        return active_subset
    
    # Sort by score (highest first)
    sorted_indices = sorted(scores.keys(), key=lambda idx: scores[idx], reverse=True)
    
    # Calculate how many to keep (top percentile, but at least min_size)
    target_size = max(min_size, int(len(active_subset) * percentile))
    
    # Keep reviewed songs (they're already in active subset)
    reviewed_in_subset = [idx for idx in active_subset if str(idx) in decisions]
    
    # Take top predictions
    top_unreviewed = sorted_indices[:max(0, target_size - len(reviewed_in_subset))]
    
    # Combine reviewed songs with top unreviewed
    new_subset = reviewed_in_subset + top_unreviewed
    
    return new_subset


def load_rows(csv_path: Path) -> List[SongRow]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"track_name", "artists", "track_genre"}
        fieldnames = set(reader.fieldnames or [])
        if not required.issubset(fieldnames):
            raise ValueError(
                f"CSV must contain columns: {sorted(required)}. Found: {sorted(fieldnames)}"
            )
        rows = [SongRow.from_dict(r) for r in reader]
    if not rows:
        raise ValueError("CSV has no rows.")
    return rows


def progress_path_for(csv_path: Path) -> Path:
    return csv_path.with_suffix(csv_path.suffix + ".progress.json")


def load_progress(progress_path: Path) -> Dict:
    if not progress_path.exists():
        return {"decisions": {}, "active_subset": None, "batch_count": 0, "excluded_categories": []}
    with progress_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "decisions" not in data:
        data["decisions"] = {}
    if "active_subset" not in data:
        data["active_subset"] = None
    if "batch_count" not in data:
        data["batch_count"] = 0
    # Support both old format (excluded_genres) and new format (excluded_categories)
    if "excluded_categories" not in data:
        if "excluded_genres" in data:
            # Convert old format: get categories from genres
            data["excluded_categories"] = []
        else:
            data["excluded_categories"] = []
    return data


def save_progress(progress_path: Path, decisions: Dict[str, str], active_subset: Optional[List[int]] = None, batch_count: int = 0, excluded_categories: Optional[List[str]] = None) -> None:
    tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump({
            "decisions": decisions,
            "active_subset": active_subset,
            "batch_count": batch_count,
            "excluded_categories": excluded_categories or []
        }, f, indent=2)
    os.replace(tmp, progress_path)


def export_lists(csv_path: Path, rows: List[SongRow], decisions: Dict[str, str]) -> Tuple[Path, Path, Path, Path]:
    rating4 = []
    rating3 = []
    rating2 = []
    rating1 = []

    for idx, row in enumerate(rows):
        rating_str = decisions.get(str(idx))
        if rating_str == "4":
            rating4.append(row)
        elif rating_str == "3":
            rating3.append(row)
        elif rating_str == "2":
            rating2.append(row)
        elif rating_str == "1":
            rating1.append(row)

    base = csv_path.with_suffix("")
    rating4_path = Path(str(base) + "_RATING4.csv")
    rating3_path = Path(str(base) + "_RATING3.csv")
    rating2_path = Path(str(base) + "_RATING2.csv")
    rating1_path = Path(str(base) + "_RATING1.csv")

    def write_out(path: Path, items: List[SongRow]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Category", "Song", "Artist", "DanceSpeed"])
            w.writeheader()
            for r in items:
                w.writerow(r.to_dict())

    write_out(rating4_path, rating4)
    write_out(rating3_path, rating3)
    write_out(rating2_path, rating2)
    write_out(rating1_path, rating1)

    return rating4_path, rating3_path, rating2_path, rating1_path


def print_song(i: int, total: int, row: SongRow, rating: Optional[str], prediction: Optional[Tuple[int, float]] = None) -> None:
    print("\n" + "-" * 70)
    print(f"[{i+1}/{total}]  Category: {row.category}   Speed: {row.dance_speed}")
    print(f"Song:   {row.song}")
    print(f"Artist: {row.artist}")
    if rating:
        print(f"Current rating: {rating}/4")
    if prediction:
        pred_rating, conf = prediction
        conf_pct = int(conf * 100)
        print(f"Predicted rating: {pred_rating}/4 (confidence: {conf_pct}%)")
    print("-" * 70)


def weighted_sample_songs(rows: List[SongRow], indices: List[int], sample_size: int) -> List[int]:
    """Sample songs with weights based on popularity and danceability (without replacement)."""
    import random
    
    if len(indices) <= sample_size:
        return list(indices)
    
    # Calculate weights for each song
    weights = []
    for idx in indices:
        row = rows[idx]
        # Popularity is 0-100, normalize to 0-1
        popularity_score = (row.popularity / 100.0) if row.popularity is not None else 0.5
        # Danceability is already 0-1
        danceability_score = row.danceability if row.danceability is not None else 0.5
        
        # Combined score (equal weight to both)
        combined_score = (popularity_score + danceability_score) / 2.0
        
        # Add a base weight to ensure all songs have some chance
        weight = combined_score + 0.1
        weights.append(weight)
    
    # Sample without replacement using weighted probabilities
    selected = []
    remaining_indices = list(indices)
    remaining_weights = list(weights)
    
    for _ in range(sample_size):
        if not remaining_indices:
            break
        
        # Normalize remaining weights
        total_weight = sum(remaining_weights)
        if total_weight == 0:
            # Fallback to uniform for remaining
            selected.extend(random.sample(remaining_indices, min(sample_size - len(selected), len(remaining_indices))))
            break
        
        # Select one based on weighted probability
        chosen_idx = random.choices(range(len(remaining_indices)), weights=remaining_weights, k=1)[0]
        selected.append(remaining_indices[chosen_idx])
        
        # Remove selected item
        remaining_indices.pop(chosen_idx)
        remaining_weights.pop(chosen_idx)
    
    return selected


def normalize_choice(raw: str) -> Tuple[str, Optional[int]]:
    raw = raw.strip()
    if not raw:
        return "", None

    if raw.lower() in {"q", "quit", "exit"}:
        return "quit", None
    if raw == "?" or raw.lower() in {"help", "h"}:
        return "help", None

    # Check for rating 1-4
    try:
        rating = int(raw)
        if 1 <= rating <= 4:
            return "rating", rating
    except ValueError:
        pass

    return "invalid", None


def main() -> None:
    import random
    
    ap = argparse.ArgumentParser(description="Progressive playlist filtering with ML assistance.")
    ap.add_argument("csv_path", type=str, nargs="?", default="dataset.csv", help="Path to the playlist CSV (Spotify dataset format). Default: dataset.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    rows = load_rows(csv_path)
    total = len(rows)
    prog_path = progress_path_for(csv_path)
    prog = load_progress(prog_path)

    decisions: Dict[str, str] = dict(prog.get("decisions", {}))
    active_subset: List[int] = prog.get("active_subset")
    batch_count = int(prog.get("batch_count", 0))
    excluded_categories: List[str] = prog.get("excluded_categories", [])
    
    # Genre selection step (only if starting fresh)
    if active_subset is None:
        # Group genres into categories
        genre_to_category = {}
        category_to_genres = {}
        for row in rows:
            if row.category:
                category = get_genre_category(row.category)
                genre_to_category[row.category] = category
                if category not in category_to_genres:
                    category_to_genres[category] = set()
                category_to_genres[category].add(row.category)
        
        # Count songs per category
        category_counts = {}
        for i, row in enumerate(rows):
            if row.category:
                category = genre_to_category[row.category]
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Sort categories by name
        all_categories = sorted(category_counts.keys())
        
        print(f"\n{'='*70}")
        print("GENRE CATEGORY SELECTION")
        print(f"{'='*70}")
        print(f"\nFound {len(all_categories)} genre categories in the dataset.")
        print("You can exclude entire categories you don't want to review.")
        print("\nCategories:")
        for i, category in enumerate(all_categories, 1):
            count = category_counts[category]
            num_genres = len(category_to_genres[category])
            print(f"  {i}. {category} ({count} songs, {num_genres} genres)")
        
        print("\nEnter category numbers to EXCLUDE (comma-separated, e.g., '1,3,5')")
        print("Or press Enter to include all categories:")
        exclude_input = input("> ").strip()
        
        excluded_categories = []
        if exclude_input:
            try:
                exclude_indices = [int(x.strip()) - 1 for x in exclude_input.split(",")]
                excluded_categories = [all_categories[i] for i in exclude_indices if 0 <= i < len(all_categories)]
                print(f"\nExcluding {len(excluded_categories)} categories: {', '.join(excluded_categories)}")
            except ValueError:
                print("Invalid input. Including all categories.")
                excluded_categories = []
        
        # Filter active subset to exclude selected categories
        # Include songs with no category in "Other" category
        active_subset = [i for i, row in enumerate(rows) 
                        if (row.category and get_genre_category(row.category) not in excluded_categories) 
                        or (not row.category and "Other" not in excluded_categories)]
        print(f"\nStarting with {len(active_subset)} songs (after category filtering).")
    else:
        rating4_count = sum(1 for v in decisions.values() if v == "4")
        print(f"\nResuming with {len(active_subset)} songs in active subset.")
        print(f"Batch {batch_count} completed. {rating4_count} songs rated 4 so far.")
        if excluded_categories:
            print(f"Excluded categories: {', '.join(excluded_categories)}")

    if not ML_AVAILABLE:
        raise RuntimeError("ML features required. Install scikit-learn: pip install scikit-learn numpy")

    # ML model setup
    model_path = model_path_for(csv_path)
    model_data: Optional[Tuple] = None
    
    print(CHOICES_HELP)
    print("\n" + "="*70)
    print("WORKFLOW: Review 15 songs per batch, then ML narrows to top 50%")
    print("Goal: Find 60 songs you rate 4 (love it)!")
    print("="*70 + "\n")

    TARGET_RATING4 = 60
    SONGS_PER_BATCH = 15
    MIN_SUBSET_SIZE = 200

    while True:
        # Check if we've reached the target
        rating4_count = sum(1 for v in decisions.values() if v == "4")
        if rating4_count >= TARGET_RATING4:
            print(f"\nCongratulations! You've rated {rating4_count} songs as 4 (love it)!")
            break

        # Get unreviewed songs from active subset
        unreviewed = [idx for idx in active_subset if str(idx) not in decisions]
        
        if not unreviewed:
            print("\nNo more unreviewed songs in current subset. Expanding search...")
            # If we've reviewed everything in subset, expand or finish
            if len(active_subset) >= total:
                print("All songs have been reviewed!")
                break
            # Could expand here, but for now just finish
            break

        # Sample up to SONGS_PER_BATCH unreviewed songs (weighted by popularity and danceability)
        batch_size = min(SONGS_PER_BATCH, len(unreviewed))
        batch_indices = weighted_sample_songs(rows, unreviewed, batch_size)
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_count + 1}: Reviewing {batch_size} songs")
        print(f"Active subset: {len(active_subset)} songs | Unreviewed: {len(unreviewed)}")
        print(f"Total 4-star ratings so far: {rating4_count}/{TARGET_RATING4}")
        print(f"{'='*70}\n")

        # Review each song in the batch
        for batch_idx, song_idx in enumerate(batch_indices):
            current = rows[song_idx]
            current_rating = decisions.get(str(song_idx))
            
            # Get ML prediction (model will be retrained after batch if needed)
            prediction: Optional[Tuple[int, float]] = None
            if model_data:
                prediction = predict_song(model_data, current)
            
            print_song(batch_idx + 1, batch_size, current, current_rating, prediction)
            print(f"(Song #{song_idx + 1} in full dataset)")

            while True:
                raw = input("Rating (1-4) or q/? : ")
                action, num = normalize_choice(raw)

                if action == "help":
                    print(CHOICES_HELP)
                    continue

                if action == "rating" and num is not None:
                    decisions[str(song_idx)] = str(num)
                    break
                elif action == "quit":
                    save_progress(prog_path, decisions, active_subset, batch_count, excluded_categories)
                    print(f"\nSaved progress to: {prog_path}")
                    rating4_path, rating3_path, rating2_path, rating1_path = export_lists(csv_path, rows, decisions)
                    print("Exported:")
                    print(f"  RATING 4: {rating4_path}")
                    print(f"  RATING 3: {rating3_path}")
                    print(f"  RATING 2: {rating2_path}")
                    print(f"  RATING 1: {rating1_path}")
                    return
                else:
                    print("Invalid input. Enter 1-4 to rate, or '?' for help.")

            # Save progress after each decision
            save_progress(prog_path, decisions, active_subset, batch_count, excluded_categories)

        batch_count += 1

        # After batch, filter active subset if we have a model
        num_decisions = sum(1 for v in decisions.values() if v in {"1", "2", "3", "4"})
        if num_decisions >= 15:
            # Retrain model with all decisions
            model_data = train_model(rows, decisions, min_samples=15)
            if model_data:
                try:
                    with model_path.open("wb") as f:
                        pickle.dump(model_data, f)
                except Exception:
                    pass
                
                print(f"\n{'='*70}")
                print("Filtering to top 50% of songs with highest predicted ratings...")
                old_size = len(active_subset)
                active_subset = filter_to_top_percentile(
                    rows, active_subset, decisions, model_data, 
                    percentile=0.5, min_size=MIN_SUBSET_SIZE
                )
                new_size = len(active_subset)
                print(f"Reduced from {old_size} to {new_size} songs in active subset.")
                print(f"{'='*70}\n")
                
                # Save updated subset
                save_progress(prog_path, decisions, active_subset, batch_count, excluded_categories)
        else:
            print(f"\n(Need {15 - num_decisions} more ratings before ML filtering starts)")

    # Final export
    save_progress(prog_path, decisions, active_subset, batch_count, excluded_categories)
    rating4_path, rating3_path, rating2_path, rating1_path = export_lists(csv_path, rows, decisions)

    rating4_count = sum(1 for v in decisions.values() if v == "4")
    rating3_count = sum(1 for v in decisions.values() if v == "3")
    rating2_count = sum(1 for v in decisions.values() if v == "2")
    rating1_count = sum(1 for v in decisions.values() if v == "1")

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Ratings: 4={rating4_count}, 3={rating3_count}, 2={rating2_count}, 1={rating1_count}")
    print("Exported:")
    print(f"  RATING 4: {rating4_path}")
    print(f"  RATING 3: {rating3_path}")
    print(f"  RATING 2: {rating2_path}")
    print(f"  RATING 1: {rating1_path}")
    print(f"Progress file (for resuming): {prog_path}")


if __name__ == "__main__":
    main()
