"""Seeded synthetic dataset generator for raw card-statement merchant descriptors.

Real card statements mangle merchant names in predictable ways: processor
prefixes (``SQ *``, ``TST*``, ``PAYPAL *``), aggressive abbreviation, store
numbers, city/state suffixes, reference digits and hard truncation at the
field width. This module builds a merchant pool across ~20 MCC-style
categories and applies those corruptions with a seeded RNG, so the same seed
always produces the same train/val/test splits.

Usage:
    python -m merchant_llm.data.generate --out data --train 6000 --val 1000 --test 1000 --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from merchant_llm.categories import CATEGORIES

# ---------------------------------------------------------------------------
# Merchant pool. Mostly invented names; a handful of well-known real brands
# are included because they dominate real statements.
# ---------------------------------------------------------------------------

MERCHANTS: dict[str, tuple[str, ...]] = {
    "grocery": (
        "Whole Foods Market",
        "Greenfield Market",
        "Harvest Lane Foods",
        "Daily Basket",
        "Corner Pantry",
        "Golden Valley Grocers",
        "Fresh Fare Supermarket",
        "Maple & Main Grocery",
        "Riverbend Foods",
        "Sunrise Market",
        "Pantry Plus",
        "Blue Crate Grocery",
        "Hillside Provisions",
    ),
    "fuel": (
        "Shell",
        "Redline Fuel Stop",
        "Interstate Petroleum",
        "QuickGas Express",
        "Summit Fuel & Go",
        "Blue Flame Gas",
        "Roadstar Fuels",
        "Canyon Gas Mart",
        "Velocity Fuel Center",
        "PetroPoint",
        "MileMark Fuel",
        "Northgate Petrol",
    ),
    "restaurants": (
        "Blue Harbor Seafood",
        "The Copper Skillet",
        "Casa Verde Cantina",
        "Old Mill Steakhouse",
        "Saffron Table",
        "Trattoria Lucca",
        "The Walnut Room",
        "Bayside Grill House",
        "Ember & Oak",
        "Lotus Garden Bistro",
        "Rustic Fork",
        "Harborview Diner",
        "Piazza Romana",
    ),
    "fast_food": (
        "Big Bun Burgers",
        "Taco Rocket",
        "Crispy Coop Chicken",
        "Slice Society Pizza",
        "Wrap Shack",
        "Golden Fry Express",
        "Dash Diner Drive Thru",
        "Sub Central",
        "Noodle Sprint",
        "Patty Palace",
        "Chick N Quick",
        "Burrito Bandit",
    ),
    "coffee": (
        "Starbucks",
        "Morning Ritual Coffee",
        "Bean Theory",
        "Copper Kettle Cafe",
        "Daily Grind Espresso",
        "Northlight Coffee Roasters",
        "Velvet Bean Cafe",
        "Brew & Co",
        "Cedar Street Coffee",
        "Espresso Junction",
        "Roast District",
        "Sunrise Beanery",
    ),
    "streaming": (
        "Netflix",
        "Spotify",
        "StreamVault",
        "CinePass Plus",
        "MelodyBox Music",
        "PixelPlay TV",
        "BingeBox",
        "AudioSphere",
        "ReelTime Streaming",
        "CloudCinema",
        "TuneWave",
        "WatchPort",
    ),
    "airlines": (
        "Delta Air Lines",
        "Meridian Airways",
        "Pacific Crest Air",
        "BlueJet Airlines",
        "Skyline Express Air",
        "Aurora Air",
        "TransGlobal Airways",
        "CoastalWing Airlines",
        "StarPath Airlines",
        "Horizon Sky Air",
        "AeroLink",
        "Cardinal Air",
    ),
    "hotels": (
        "Grandview Hotel & Suites",
        "Lakeside Inn",
        "The Beacon Hotel",
        "Palm Court Resort",
        "Ironwood Lodge",
        "Cityline Suites",
        "Harbor Gate Hotel",
        "Stonebridge Hotel",
        "The Juniper House",
        "Bayfront Plaza Hotel",
        "Alpine Meadow Lodge",
        "Crescent Bay Resort",
    ),
    "rideshare": (
        "Uber",
        "Lyft",
        "GlideRide",
        "CityHopper Rides",
        "ZipCab",
        "MetroDash",
        "SwiftSeat",
        "RideLoop",
        "UrbanShuttle Co",
        "FleetFox",
        "HailNow",
        "VelocityRides",
    ),
    "transit": (
        "Metro Transit Authority",
        "City Rail Pass",
        "BayLink Ferries",
        "Northline Commuter Rail",
        "Downtown Bus Co",
        "RapidWay Transit",
        "Harbor City Metro",
        "GreenLine Trams",
        "Crosstown Transit",
        "Valley Express Buses",
        "CapitalWay Metro",
        "SeasidePass Transit",
    ),
    "pharmacy": (
        "Walgreens",
        "Wellspring Pharmacy",
        "MedPoint Drugs",
        "CareFirst Apothecary",
        "GreenCross Pharmacy",
        "Hometown Rx",
        "VitalCare Drugstore",
        "Summit Health Pharmacy",
        "QuickScript Pharmacy",
        "Lakeview Drugs",
        "Cornerstone Rx",
        "Beacon Pharmacy",
    ),
    "electronics": (
        "Voltage Electronics",
        "ByteWorks Computers",
        "Nova Tech Outlet",
        "GigaHub Electronics",
        "Pixel & Wire",
        "TechBay Superstore",
        "Quantum Gadgets",
        "SoundStage Audio",
        "CoreLogic Computers",
        "Bright Screen TV Co",
        "Gadget Garage",
        "FuseBox Electronics",
    ),
    "clothing": (
        "Thread & Needle",
        "Urban Stitch Apparel",
        "Cedar Closet Clothing",
        "Maple Row Outfitters",
        "True North Denim",
        "Velvet Hanger Boutique",
        "Coastline Apparel",
        "Stitchcraft Clothing Co",
        "The Wardrobe Works",
        "Loom & Label",
        "Bold Thread Menswear",
        "Summit Outfitters",
    ),
    "home_improvement": (
        "ToolShed Hardware",
        "Keystone Home Center",
        "BuildRight Supply",
        "Hammer & Nail Co",
        "Cedar Ridge Lumber",
        "ProFix Hardware",
        "HouseWorks Depot",
        "Anchor Bolt Supply",
        "GreenThumb Garden & Home",
        "Precision Paint & Tile",
        "Oakline Building Supply",
        "Granite Peak Home Store",
    ),
    "telecom": (
        "Vertex Wireless",
        "ClearWave Mobile",
        "TalkPoint Telecom",
        "SignalOne Communications",
        "BlueSky Broadband",
        "MetroConnect Wireless",
        "PulseNet Mobile",
        "LinkLine Telecom",
        "AirGrid Communications",
        "NovaCell Wireless",
        "BeamPath Internet",
        "EchoTel Mobile",
    ),
    "utilities": (
        "City Power & Light",
        "Riverside Water Utility",
        "Northern Gas Works",
        "Evergreen Energy Co",
        "Municipal Water District",
        "ClearFlow Water Services",
        "Summit Electric Cooperative",
        "Lakeshore Utilities",
        "GridPoint Energy",
        "ValleyGas Utility",
        "Beacon Power Company",
        "BlueRiver Hydro",
    ),
    "fitness": (
        "IronWorks Gym",
        "FlexCity Fitness",
        "Summit Peak Climbing",
        "PulsePoint Studio",
        "GreenMat Yoga",
        "Titan Strength Club",
        "CardioLab Fitness",
        "The Sweat Society",
        "PowerHouse Athletics",
        "ZenFlow Pilates",
        "Apex Performance Gym",
        "StrideFit Running Club",
    ),
    "entertainment": (
        "Starlight Cinemas",
        "Arcade Alley",
        "Grand Stage Theater",
        "LaserZone FunPark",
        "Comedy Corner Club",
        "Bowl-O-Rama Lanes",
        "Adventure Cove Park",
        "Velvet Curtain Playhouse",
        "GameOn Esports Lounge",
        "Harborview Aquarium",
        "Pinball Palace",
        "Rialto Movie House",
    ),
    "online_retail": (
        "Amazon",
        "ShopSphere",
        "CartWheel Online",
        "BoxDrop Marketplace",
        "ClickCrate",
        "DailyDeal Depot",
        "ParcelPoint Online",
        "NetBasket",
        "EverShop Direct",
        "QuickPick Outlet",
        "OrderOwl",
        "SwiftCart",
    ),
    "insurance": (
        "SafeHarbor Insurance",
        "BlueRock Auto Insurance",
        "Evergreen Life Insurance",
        "Anchor Mutual Insurance",
        "ClearPath Insurance Group",
        "GateGuard Insurance",
        "HomeSure Insurance",
        "TrueNorth Assurance",
        "Cornerstone Casualty",
        "PrimeCover Insurance",
        "Sentinel Insurance Co",
        "Harbor Light Insurance",
    ),
}

PROCESSOR_PREFIXES = ("SQ *", "TST* ", "PAYPAL *", "PP*", "IN *", "PY *", "GOOGLE *")

CITIES = (
    ("NEW YORK", "NY"),
    ("BROOKLYN", "NY"),
    ("LOS ANGELES", "CA"),
    ("SAN FRANCISCO", "CA"),
    ("SAN JOSE", "CA"),
    ("CHICAGO", "IL"),
    ("HOUSTON", "TX"),
    ("AUSTIN", "TX"),
    ("DALLAS", "TX"),
    ("PHOENIX", "AZ"),
    ("PHILADELPHIA", "PA"),
    ("SAN ANTONIO", "TX"),
    ("SAN DIEGO", "CA"),
    ("SEATTLE", "WA"),
    ("DENVER", "CO"),
    ("BOSTON", "MA"),
    ("NASHVILLE", "TN"),
    ("PORTLAND", "OR"),
    ("LAS VEGAS", "NV"),
    ("MIAMI", "FL"),
    ("ORLANDO", "FL"),
    ("TAMPA", "FL"),
    ("ATLANTA", "GA"),
    ("MINNEAPOLIS", "MN"),
    ("DETROIT", "MI"),
    ("CHARLOTTE", "NC"),
    ("RALEIGH", "NC"),
    ("COLUMBUS", "OH"),
    ("CLEVELAND", "OH"),
    ("KANSAS CITY", "MO"),
    ("ST LOUIS", "MO"),
    ("SALT LAKE CITY", "UT"),
    ("PITTSBURGH", "PA"),
    ("BALTIMORE", "MD"),
    ("MILWAUKEE", "WI"),
)

# Common statement abbreviations applied before the generic vowel-stripping rule.
KNOWN_ABBREVIATIONS = {
    "market": "MKT",
    "center": "CTR",
    "company": "CO",
    "restaurant": "REST",
    "insurance": "INS",
    "pharmacy": "PHARM",
    "communications": "COMM",
    "electronics": "ELECTR",
    "supermarket": "SUPRMKT",
    "grocery": "GROC",
    "hardware": "HDWE",
    "services": "SVCS",
    "express": "EXPR",
    "airlines": "AIR",
    "airways": "AIRWYS",
    "streaming": "STREAM",
    "coffee": "COFFEE",
    "fitness": "FITNS",
    "authority": "AUTH",
    "district": "DIST",
    "cooperative": "COOP",
    "boutique": "BTQ",
    "apparel": "APPRL",
    "marketplace": "MKTPL",
}

STATEMENT_WIDTHS = (19, 22, 24, 25, 26)

_VOWELS = set("aeiouAEIOU")


def _strip_vowels(word: str) -> str:
    """Drop interior vowels the way statement processors abbreviate words."""
    if len(word) <= 4:
        return word
    head, tail = word[0], word[1:]
    stripped = head + "".join(ch for ch in tail if ch not in _VOWELS)
    return stripped if len(stripped) >= 3 else word[:4]


def _abbreviate_word(word: str, rng: random.Random) -> str:
    lower = word.lower()
    if lower in KNOWN_ABBREVIATIONS and rng.random() < 0.7:
        return KNOWN_ABBREVIATIONS[lower]
    roll = rng.random()
    if roll < 0.45:
        return _strip_vowels(word)
    if roll < 0.75:
        return word[: rng.randint(3, 5)]
    return word


def _mangle_name(name: str, rng: random.Random) -> str:
    """Turn a canonical merchant name into a statement-style token sequence."""
    words = name.replace("&", rng.choice(["&", "AND", "&"])).split()
    if words and words[0].lower() == "the" and rng.random() < 0.6:
        words = words[1:]
    if len(words) > 2 and rng.random() < 0.35:
        words = words[: rng.randint(2, len(words))]
    out = []
    for word in words:
        if word in {"&", "AND"} or len(word) <= 2:
            out.append(word)
        elif rng.random() < 0.45:
            out.append(_abbreviate_word(word, rng))
        else:
            out.append(word)
    joiner = rng.choice([" ", " ", " ", "*", " * ", "-", ""])
    return joiner.join(out)


def _store_number(rng: random.Random) -> str:
    style = rng.random()
    digits = rng.randint(1, 99999)
    if style < 0.5:
        return f"#{digits:0{rng.choice([3, 4, 5])}d}"
    if style < 0.75:
        return f"STORE {digits % 1000}"
    return f"{rng.randint(100, 9999)}"


def _reference_digits(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(8, 11)))


def _phone_fragment(rng: random.Random) -> str:
    return f"8{rng.choice(['00', '44', '66', '77', '88'])}-555-{rng.randint(0, 9999):04d}"


def _apply_case(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.78:
        return text.upper()
    if roll < 0.9:
        return text
    return "".join(ch.upper() if rng.random() < 0.5 else ch.lower() for ch in text)


def make_raw_string(name: str, category: str, rng: random.Random) -> str:
    """Build one corrupted statement descriptor for a canonical merchant name."""
    parts: list[str] = []

    prefix = ""
    if rng.random() < 0.28:
        prefix = rng.choice(PROCESSOR_PREFIXES)

    mangled = _mangle_name(name, rng)
    parts.append(mangled)

    if rng.random() < 0.32:
        parts.append(_store_number(rng))
    if rng.random() < 0.14:
        parts.append(_reference_digits(rng))
    if rng.random() < 0.08:
        parts.append(_phone_fragment(rng))
    if rng.random() < 0.10:
        parts[-1] = parts[-1] + ".COM"
    if rng.random() < 0.40:
        city, state = rng.choice(CITIES)
        parts.append(f"{city} {state}" if rng.random() < 0.8 else city)

    sep = rng.choice([" ", " ", "  ", " * "])
    raw = prefix + sep.join(p for p in parts if p)

    raw = _apply_case(raw, rng)

    if rng.random() < 0.35:
        raw = raw[: rng.choice(STATEMENT_WIDTHS)].rstrip()

    # Occasional stray double spaces survive on real statements.
    if rng.random() < 0.08:
        idx = rng.randint(0, max(0, len(raw) - 1))
        raw = raw[:idx] + " " + raw[idx:]

    return raw.strip()


def generate_examples(n: int, seed: int) -> list[dict[str, str]]:
    """Generate ``n`` unique examples. Deterministic for a given seed."""
    rng = random.Random(seed)
    pool = [(name, category) for category, names in MERCHANTS.items() for name in names]
    seen: set[str] = set()
    examples: list[dict[str, str]] = []
    attempts = 0
    max_attempts = n * 50
    while len(examples) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(f"could not generate {n} unique raw strings (got {len(examples)})")
        name, category = rng.choice(pool)
        raw = make_raw_string(name, category, rng)
        if len(raw) < 4 or raw in seen:
            continue
        seen.add(raw)
        examples.append({"raw": raw, "merchant_name": name, "category": category})
    return examples


def generate_splits(
    train: int, val: int, test: int, seed: int
) -> dict[str, list[dict[str, str]]]:
    """Generate disjoint train/val/test splits (unique raw strings across all)."""
    total = generate_examples(train + val + test, seed)
    rng = random.Random(seed + 1)
    rng.shuffle(total)
    return {
        "train": total[:train],
        "val": total[train : train + val],
        "test": total[train + val :],
    }


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--val", type=int, default=1000)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="rows copied from the train split into sample.jsonl (committed to git)",
    )
    args = parser.parse_args(argv)

    splits = generate_splits(args.train, args.val, args.test, args.seed)
    for split_name, rows in splits.items():
        write_jsonl(args.out / f"{split_name}.jsonl", rows)
        print(f"wrote {len(rows):>5} rows -> {args.out / f'{split_name}.jsonl'}")
    write_jsonl(args.out / "sample.jsonl", splits["train"][: args.sample_size])

    meta = {
        "seed": args.seed,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "categories": list(CATEGORIES),
        "merchant_pool_size": sum(len(v) for v in MERCHANTS.values()),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote metadata -> {args.out / 'meta.json'}")


if __name__ == "__main__":
    main()
