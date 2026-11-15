
# ml/aspect_config.py

"""
Aspect taxonomy for the Customer Review Intelligence project.

Cluster IDs provide initial hints about which aspects
a review may discuss.

Keywords and phrases will help identify aspect mentions
inside individual reviews.
"""

ASPECTS = {

    "sound_quality": {
        "name": "Sound Quality",
        "source_clusters": [6, 9, 17],
        "keywords": [
            "sound quality",
            "audio quality",
            "bass",
            "deep bass",
            "treble",
            "sound",
            "music quality",
            "clear sound",
            "audio",
            "volume"
        ]
    },

    "comfort_fit": {
        "name": "Comfort and Ear Fit",
        "source_clusters": [8, 18],
        "keywords": [
            "comfortable",
            "uncomfortable",
            "comfort",
            "ear fit",
            "fit my ears",
            "fits well",
            "earbuds fit",
            "hurt my ears",
            "pain in ears",
            "ear pieces"
        ]
    },

    "workout_stability": {
        "name": "Workout Stability",
        "source_clusters": [21],
        "keywords": [
            "running",
            "workout",
            "gym",
            "exercise",
            "stay in place",
            "stay in my ears",
            "fall out",
            "falling out",
            "treadmill",
            "jogging"
        ]
    },

    "battery": {
        "name": "Battery Performance",
        "source_clusters": [9, 14],
        "keywords": [
            "battery life",
            "battery",
            "battery drains",
            "battery lasts",
            "battery backup",
            "playback time",
            "hours of battery",
            "battery performance",
            "battery dies",
            "battery charge"
        ]
    },

    "charging_durability": {
        "name": "Charging and Durability",
        "source_clusters": [4, 14, 15],
        "keywords": [
            "charging",
            "charging port",
            "charger port",
            "stopped charging",
            "won't charge",
            "not charging",
            "holding a charge",
            "charging case",
            "stopped working",
            "broke",
            "broken",
            "stopped working after",
            "durability",
            "lasted"
        ]
    },

    "connectivity": {
        "name": "Bluetooth Connectivity",
        "source_clusters": [5, 11, 17],
        "keywords": [
            "bluetooth",
            "connection",
            "connectivity",
            "disconnect",
            "disconnecting",
            "keeps disconnecting",
            "signal drops",
            "pairing",
            "pair",
            "connect",
            "connection issues",
            "cuts out"
        ]
    },

    "noise_cancellation": {
        "name": "Noise Cancellation",
        "source_clusters": [7],
        "keywords": [
            "noise cancellation",
            "noise cancelling",
            "noise canceling",
            "noise isolation",
            "background noise",
            "blocks noise",
            "ambient noise"
        ]
    },

    "water_resistance": {
        "name": "Water and Sweat Resistance",
        "source_clusters": [10, 13],
        "keywords": [
            "waterproof",
            "water resistant",
            "water resistance",
            "sweat resistant",
            "sweat resistance",
            "sweatproof",
            "sweat",
            "water",
            "swimming",
            "pool"
        ]
    },

    "audio_defects": {
        "name": "Audio and Earbud Defects",
        "source_clusters": [12],
        "keywords": [
            "one earbud",
            "one ear",
            "left earbud",
            "right earbud",
            "left side",
            "right side",
            "only one side",
            "one side works",
            "no sound",
            "no audio",
            "sound stopped",
            "earbud not working"
        ]
    },

    "price_value": {
        "name": "Price and Value for Money",
        "source_clusters": [0, 2, 19],
        "keywords": [
            "price",
            "value for money",
            "worth the money",
            "worth the price",
            "affordable",
            "inexpensive",
            "expensive",
            "cheap",
            "price point",
            "reasonable price"
        ]
    },

    "design_build": {
        "name": "Design and Build Quality",
        "source_clusters": [20],
        "keywords": [
            "design",
            "build quality",
            "flexible",
            "flexibility",
            "ear hook",
            "hook design",
            "wireless design",
            "material",
            "plastic quality",
            "construction"
        ]
    },

    "accessories": {
        "name": "Accessories and Packaging",
        "source_clusters": [16],
        "keywords": [
            "carrying case",
            "storage case",
            "ear tips",
            "foam tips",
            "extra tips",
            "accessories",
            "packaging",
            "package",
            "box",
            "earbud case"
        ]
    }
}


# Cluster IDs that contain purchasing-related insights.
# These will be handled separately from product aspects.

PURCHASE_INSIGHTS = {

    "gift_purchases": {
        "name": "Gift Purchases",
        "source_clusters": [1],
        "keywords": [
            "gift",
            "birthday",
            "bought for my husband",
            "bought for my wife",
            "gift for"
        ]
    },

    "repeat_purchases": {
        "name": "Repeat Purchases and Loyalty",
        "source_clusters": [3],
        "keywords": [
            "second pair",
            "third pair",
            "bought another",
            "buy again",
            "purchased again",
            "repeat purchase"
        ]
    }
}
