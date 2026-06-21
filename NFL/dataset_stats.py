import os
import csv

DATASET_DIR = "/Users/aly/Documents/University_of_Waterloo/Winter 2025/Research/dataset_backup"
TRAIN_YEARS = list(range(2016, 2024))
TEST_YEARS = [2024, 2025]


def count_rows(path):
    with open(path, "r") as f:
        return sum(1 for _ in f) - 1  # subtract header


def stats_for_year(year):
    year_dir = os.path.join(DATASET_DIR, str(year))
    if not os.path.isdir(year_dir):
        return None
    files = [f for f in os.listdir(year_dir) if f.endswith(".csv")]
    if not files:
        return None
    counts = [count_rows(os.path.join(year_dir, f)) for f in files]
    return {
        "year": year,
        "games": len(counts),
        "events": sum(counts),
        "max_events": max(counts),
        "min_events": min(counts),
        "avg_events": sum(counts) / len(counts),
    }


def aggregate(year_stats_list):
    total_games = sum(s["games"] for s in year_stats_list)
    total_events = sum(s["events"] for s in year_stats_list)
    return {
        "games": total_games,
        "events": total_events,
        "max_events": max(s["max_events"] for s in year_stats_list),
        "min_events": min(s["min_events"] for s in year_stats_list),
        "avg_events": total_events / total_games,
    }


def print_row(label, s, is_year=True):
    if is_year:
        print(f"  {label:<6}  games={s['games']:>4}  events={s['events']:>7}  "
              f"max={s['max_events']:>4}  min={s['min_events']:>4}  avg={s['avg_events']:>7.1f}")
    else:
        print(f"  {label:<10}  games={s['games']:>4}  events={s['events']:>7}  "
              f"max={s['max_events']:>4}  min={s['min_events']:>4}  avg={s['avg_events']:>7.1f}")


print("=" * 65)
print("Per-year breakdown")
print("=" * 65)

train_stats, test_stats = [], []

for year in TRAIN_YEARS + TEST_YEARS:
    s = stats_for_year(year)
    if s is None:
        print(f"  {year}  — not found, skipping")
        continue
    print_row(str(year), s)
    if year in TRAIN_YEARS:
        train_stats.append(s)
    else:
        test_stats.append(s)

print()
print("=" * 65)
print("Aggregated")
print("=" * 65)
if train_stats:
    print_row("Train (2016-2023)", aggregate(train_stats), is_year=False)
if test_stats:
    print_row("Test  (2024-2025)", aggregate(test_stats), is_year=False)
