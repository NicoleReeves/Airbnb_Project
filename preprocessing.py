"""
Complete preprocessing pipeline for Airbnb price prediction
"""

import pandas as pd
import numpy as np
import re

# Amenity Categories

basic_amenities = {
    'wifi': ['wifi', 'wi-fi', 'wireless internet', 'internet'],
    'kitchen': ['kitchen', 'kitchenette', 'full kitchen'],
    'air_conditioning': ['air conditioning', 'ac', 'air con', 'cooling'],
    'heating': ['heating', 'heater', 'central heating', 'radiator'],
    'tv': ['tv', 'television', 'cable tv', 'smart tv', 'netflix'],
    'washer': ['washer', 'washing machine', 'laundry'],
    'dryer': ['dryer', 'tumble dryer', 'drying machine'],
    'iron': ['iron', 'ironing board'],
    'hair_dryer': ['hair dryer', 'hairdryer', 'blow dryer'],
    'essentials': ['essentials', 'basics', 'towels', 'bed sheets', 'soap', 'toilet paper']
}

safety_amenities = {
    'smoke_alarm': ['smoke alarm', 'smoke detector', 'fire alarm'],
    'carbon_monoxide_alarm': ['carbon monoxide alarm', 'co detector', 'carbon monoxide detector'],
    'first_aid_kit': ['first aid kit', 'medical kit'],
    'fire_extinguisher': ['fire extinguisher'],
    'security_cameras': ['security cameras', 'surveillance', 'cctv'],
    'lockbox': ['lockbox', 'key safe', 'lock box'],
    'private_entrance': ['private entrance', 'separate entrance', 'own entrance']
}

kitchen_dining = {
    'refrigerator': ['refrigerator', 'fridge', 'mini fridge', 'mini-fridge'],
    'microwave': ['microwave', 'micro wave'],
    'oven': ['oven', 'stove', 'cooktop', 'hob'],
    'dishwasher': ['dishwasher', 'dish washer'],
    'coffee_maker': ['coffee maker', 'coffee machine', 'espresso machine', 'nespresso'],
    'dining_table': ['dining table', 'dining area', 'eating area'],
    'cookware': ['cooking basics', 'pots and pans', 'cookware', 'dishes and silverware'],
    'blender': ['blender', 'food processor'],
    'toaster': ['toaster'],
    'kettle': ['kettle', 'electric kettle']
}

bathroom_amenities = {
    'shampoo': ['shampoo', 'body soap', 'shower gel'],
    'conditioner': ['conditioner'],
    'body_soap': ['body soap', 'soap', 'shower gel'],
    'hot_water': ['hot water', 'hot shower'],
    'bathtub': ['bathtub', 'bath tub', 'bath'],
    'bidet': ['bidet'],
    'bathroom_essentials': ['bathroom essentials', 'bath towels', 'toilet paper']
}

bedroom_living = {
    'bed_linens': ['bed linens', 'bedding', 'sheets', 'pillows'],
    'extra_pillows': ['extra pillows', 'pillows and blankets'],
    'hangers': ['hangers', 'coat hangers', 'wardrobe'],
    'closet': ['closet', 'wardrobe', 'clothing storage'],
    'desk': ['desk', 'workspace', 'laptop friendly workspace'],
    'chair': ['chair', 'office chair', 'desk chair'],
    'sofa': ['sofa', 'couch', 'living room'],
    'blackout_curtains': ['blackout curtains', 'room darkening shades']
}

internet_office = {
    'dedicated_workspace': ['dedicated workspace', 'office space', 'work area'],
    'laptop_friendly': ['laptop friendly', 'laptop workspace'],
    'ethernet': ['ethernet connection', 'wired internet'],
    'printer': ['printer'],
    'monitor': ['monitor', 'external monitor']
}

entertainment = {
    'sound_system': ['sound system', 'speakers', 'stereo'],
    'game_console': ['game console', 'playstation', 'xbox', 'nintendo'],
    'books': ['books', 'reading material'],
    'board_games': ['board games', 'card games', 'games'],
    'music': ['music', 'spotify', 'streaming']
}

outdoor_recreation = {
    'balcony': ['balcony', 'terrace', 'patio'],
    'garden': ['garden', 'yard', 'outdoor space'],
    'bbq_grill': ['bbq grill', 'barbecue', 'grill', 'outdoor grill'],
    'outdoor_furniture': ['outdoor furniture', 'patio furniture', 'garden furniture'],
    'beach_access': ['beach access', 'beachfront', 'waterfront'],
    'mountain_view': ['mountain view', 'mountains'],
    'city_view': ['city view', 'skyline view'],
    'garden_view': ['garden view', 'park view']
}

luxury_amenities = {
    'pool': ['pool', 'swimming pool', 'shared pool', 'private pool'],
    'hot_tub': ['hot tub', 'jacuzzi', 'spa'],
    'gym': ['gym', 'fitness centre', 'exercise equipment', 'weights'],
    'sauna': ['sauna', 'steam room'],
    'concierge': ['concierge', 'doorman', 'reception'],
    'room_service': ['room service', 'housekeeping'],
    'luxury_toiletries': ['luxury toiletries', 'premium amenities'],
    'wine_cooler': ['wine cooler', 'wine fridge', 'mini bar']
}

transport_location = {
    'free_parking': ['free parking', 'parking included', 'garage'],
    'paid_parking': ['paid parking', 'parking available'],
    'ev_charger': ['ev charger', 'electric vehicle charging', 'tesla charger'],
    'public_transport': ['near public transport', 'metro', 'subway access'],
    'bicycle': ['bicycle', 'bike', 'cycling'],
    'airport_shuttle': ['airport shuttle', 'transfer service']
}

family_accessibility = {
    'family_friendly': ['family friendly', 'child friendly', 'kids welcome'],
    'crib': ['crib', 'baby cot', 'cot'],
    'high_chair': ['high chair', 'baby chair'],
    'baby_bath': ['baby bath', 'bathtub for babies'],
    'step_free_access': ['step free access', 'wheelchair accessible', 'accessible'],
    'wide_doorways': ['wide doorways', 'accessible doorways'],
    'accessible_bathroom': ['accessible bathroom', 'roll-in shower']
}

pet_amenities = {
    'pets_allowed': ['pets allowed', 'pet friendly', 'dogs allowed', 'cats allowed'],
    'pet_bowls': ['pet bowls', 'dog bowls'],
    'pet_bed': ['pet bed', 'dog bed']
}

climate_environment = {
    'fan': ['fan', 'ceiling fan', 'portable fan'],
    'fireplace': ['fireplace', 'wood burning fireplace'],
    'humidifier': ['humidifier'],
    'air_purifier': ['air purifier', 'hepa filter'],
    'mosquito_net': ['mosquito net', 'bug net']
}

all_amenity_categories = {
    'Basic': basic_amenities,
    'Safety': safety_amenities,
    'Kitchen_Dining': kitchen_dining,
    'Bathroom': bathroom_amenities,
    'Bedroom_Living': bedroom_living,
    'Internet_Office': internet_office,
    'Entertainment': entertainment,
    'Outdoor_Recreation': outdoor_recreation,
    'Luxury': luxury_amenities,
    'Transport_Location': transport_location,
    'Family_Accessibility': family_accessibility,
    'Pet': pet_amenities,
    'Climate_Environment': climate_environment
}

# Text analysis functions


def calculate_readability_score(text):
    """Calculate readability score using Flesch formula"""
    if pd.isna(text) or len(str(text).strip()) == 0:
        return 0
    
    text_str = str(text)
    sentences = len([s for s in text_str.split('.') if s.strip()])
    words = len(text_str.split())
    
    if sentences == 0 or words == 0:
        return 0
    
    vowels = 'aeiouAEIOU'
    syllables = sum(1 for char in text_str if char in vowels)
    
    if syllables == 0:
        syllables = words
    
    avg_sentence_length = words / sentences
    avg_syllables_per_word = syllables / words
    
    readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    return max(0, min(100, readability))

def calculate_sentiment_score(text):
    """Calculate sentiment"""
    if pd.isna(text):
        return 0
    
    text_lower = str(text).lower()
    
    positive_words = [
        'amazing', 'beautiful', 'perfect', 'excellent', 'wonderful', 'fantastic', 
        'great', 'awesome', 'lovely', 'stunning', 'spectacular', 'incredible',
        'comfortable', 'cosy', 'cozy', 'charming', 'peaceful', 'relaxing', 
        'enjoyable', 'delightful', 'convenient', 'spacious', 'bright', 'clean', 
        'modern', 'stylish', 'elegant', 'sophisticated', 'luxury', 'premium',
        'superb', 'outstanding', 'exceptional', 'brilliant', 'magnificent',
        'gorgeous', 'fabulous', 'splendid', 'marvellous', 'marvelous'
    ]
    
    negative_words = [
        'terrible', 'awful', 'bad', 'horrible', 'disappointing', 'dirty',
        'noisy', 'uncomfortable', 'small', 'cramped', 'old', 'outdated',
        'inconvenient', 'difficult', 'problems', 'issues', 'broken',
        'poor', 'worst', 'unpleasant', 'disgusting', 'nasty', 'dreadful'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text_lower.split())
    if total_words == 0:
        return 0
    
    sentiment_score = (positive_count - negative_count) / max(total_words / 20, 1)
    return max(-1, min(1, sentiment_score))

def extract_description_features(description):
    """Extract features from description"""
    if pd.isna(description):
        return {
            'desc_length': 0, 'desc_word_count': 0, 'desc_sentence_count': 0,
            'avg_word_length': 0, 'desc_readability': 0, 'desc_sentiment_score': 0,
            'desc_luxury_mentions': 0, 'desc_location_mentions': 0, 'desc_transport_mentions': 0,
            'desc_experience_mentions': 0, 'desc_facility_mentions': 0, 'desc_business_mentions': 0,
            'desc_safety_mentions': 0, 'desc_cleanliness_mentions': 0, 'desc_comfort_mentions': 0,
            'desc_view_mentions': 0, 'desc_activity_mentions': 0, 'desc_food_mentions': 0,
            'desc_family_mentions': 0, 'desc_romantic_mentions': 0, 'desc_exclamation_count': 0,
            'desc_question_count': 0, 'desc_caps_ratio': 0, 'desc_number_count': 0,
            'desc_char_diversity': 0,
            'desc_luxury_themes_score': 0, 'desc_location_themes_score': 0,
            'desc_experience_themes_score': 0, 'desc_amenity_themes_score': 0,
            'desc_comfort_themes_score': 0, 'desc_space_themes_score': 0,
            'desc_emotional_score': 0, 'desc_urgency_score': 0, 'desc_cleanliness_score': 0,
            'desc_business_score': 0
        }
    
    desc_str = str(description)
    desc_lower = desc_str.lower()
    words = desc_str.split()
    
    features = {}
    
    # Basic text statistics
    features['desc_length'] = len(desc_str)
    features['desc_word_count'] = len(words)
    features['desc_sentence_count'] = len([s for s in desc_str.split('.') if s.strip()])
    features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
    
    # Character diversity
    unique_chars = len(set(desc_str.lower()))
    features['desc_char_diversity'] = unique_chars / len(desc_str) if len(desc_str) > 0 else 0
    
    # Readability and sentiment
    features['desc_readability'] = calculate_readability_score(desc_str)
    features['desc_sentiment_score'] = calculate_sentiment_score(desc_str)
    
    # Theme-based mentions
    luxury_words = ['luxury', 'luxurious', 'premium', 'upscale', 'high-end', 'exclusive', 'elegant']
    features['desc_luxury_mentions'] = sum(1 for word in luxury_words if word in desc_lower)
    
    location_words = ['location', 'neighbourhood', 'neighborhood', 'area', 'district', 'zone',
                     'close', 'near', 'walking', 'minutes', 'central', 'convenient']
    features['desc_location_mentions'] = sum(1 for word in location_words if word in desc_lower)
    
    transport_words = ['metro', 'tube', 'underground', 'subway', 'bus', 'train', 'station', 
                      'transport', 'uber', 'taxi', 'airport', 'railway']
    features['desc_transport_mentions'] = sum(1 for word in transport_words if word in desc_lower)
    
    experience_words = ['experience', 'enjoy', 'relax', 'explore', 'discover', 'adventure', 
                       'stay', 'visit', 'holiday', 'vacation', 'getaway']
    features['desc_experience_mentions'] = sum(1 for word in experience_words if word in desc_lower)
    
    facility_words = ['kitchen', 'bathroom', 'bedroom', 'living', 'dining', 'balcony', 
                     'garden', 'parking', 'wifi', 'pool', 'gym']
    features['desc_facility_mentions'] = sum(1 for word in facility_words if word in desc_lower)
    
    business_words = ['business', 'work', 'workspace', 'office', 'meetings', 'conference', 
                     'professional', 'corporate']
    features['desc_business_mentions'] = sum(1 for word in business_words if word in desc_lower)
    
    safety_words = ['safe', 'secure', 'security', 'safety', 'protected', 'gated', 'keyless']
    features['desc_safety_mentions'] = sum(1 for word in safety_words if word in desc_lower)
    
    cleanliness_words = ['clean', 'fresh', 'spotless', 'sanitised', 'sanitized', 'hygienic', 'tidy']
    features['desc_cleanliness_mentions'] = sum(1 for word in cleanliness_words if word in desc_lower)
    
    comfort_words = ['comfortable', 'cosy', 'cozy', 'relaxing', 'peaceful', 'quiet', 'serene']
    features['desc_comfort_mentions'] = sum(1 for word in comfort_words if word in desc_lower)
    
    view_words = ['view', 'views', 'overlook', 'facing', 'panoramic', 'scenic']
    features['desc_view_mentions'] = sum(1 for word in view_words if word in desc_lower)
    
    activity_words = ['restaurant', 'shopping', 'museum', 'theatre', 'theater', 'park', 'beach', 
                     'nightlife', 'entertainment', 'attractions']
    features['desc_activity_mentions'] = sum(1 for word in activity_words if word in desc_lower)
    
    food_words = ['restaurant', 'food', 'dining', 'cafe', 'coffee', 'breakfast', 'kitchen']
    features['desc_food_mentions'] = sum(1 for word in food_words if word in desc_lower)
    
    family_words = ['family', 'children', 'kids', 'child-friendly', 'family-friendly']
    features['desc_family_mentions'] = sum(1 for word in family_words if word in desc_lower)
    
    romantic_words = ['romantic', 'couple', 'honeymoon', 'intimate', 'private']
    features['desc_romantic_mentions'] = sum(1 for word in romantic_words if word in desc_lower)
    
    # Punctuation and formatting
    features['desc_exclamation_count'] = desc_str.count('!')
    features['desc_question_count'] = desc_str.count('?')
    
    caps_count = sum(1 for char in desc_str if char.isupper())
    features['desc_caps_ratio'] = caps_count / len(desc_str) if len(desc_str) > 0 else 0
    
    features['desc_number_count'] = sum(1 for word in words if any(char.isdigit() for char in word))
    
    # Theme scores (aggregated)
    features['desc_luxury_themes_score'] = features['desc_luxury_mentions'] * 2
    features['desc_location_themes_score'] = features['desc_location_mentions'] + features['desc_transport_mentions']
    features['desc_experience_themes_score'] = features['desc_experience_mentions']
    features['desc_amenity_themes_score'] = features['desc_facility_mentions']
    features['desc_comfort_themes_score'] = features['desc_comfort_mentions']
    features['desc_space_themes_score'] = features['desc_view_mentions']
    features['desc_emotional_score'] = features['desc_sentiment_score'] * 10
    features['desc_urgency_score'] = features['desc_exclamation_count']
    features['desc_cleanliness_score'] = features['desc_cleanliness_mentions'] * 2
    features['desc_business_score'] = features['desc_business_mentions'] * 2
    
    return features

def extract_name_features(name):
    """Extract features from listing name"""
    if pd.isna(name):
        return {
            'name_length': 0, 'name_word_count': 0, 'name_luxury_score': 0,
            'name_location_score': 0, 'name_mentions_apartment': False,
            'name_mentions_house': False, 'name_mentions_studio': False,
            'name_mentions_loft': False, 'name_mentions_room': False,
            'name_comfort_score': 0, 'name_mentions_private': False,
            'name_mentions_entire': False, 'name_view_score': 0,
            'name_mentions_central': False, 'name_mentions_modern': False
        }
    
    name_lower = str(name).lower()
    features = {}
    
    features['name_length'] = len(name)
    features['name_word_count'] = len(name.split())
    
    luxury_words = ['luxury', 'luxurious', 'premium', 'deluxe', 'executive', 
                   'penthouse', 'villa', 'mansion', 'suite', 'presidential']
    features['name_luxury_score'] = sum(1 for word in luxury_words if word in name_lower)
    
    location_words = ['central', 'centre', 'center', 'downtown', 'city centre', 
                     'city center', 'heart of', 'near', 'close to', 'walking distance',
                     'zone 1', 'zone 2', 'prime location']
    features['name_location_score'] = sum(1 for word in location_words if word in name_lower)
    
    features['name_mentions_apartment'] = any(word in name_lower for word in ['apartment', 'flat', 'apt'])
    features['name_mentions_house'] = any(word in name_lower for word in ['house', 'home', 'cottage', 'townhouse'])
    features['name_mentions_studio'] = 'studio' in name_lower
    features['name_mentions_loft'] = 'loft' in name_lower
    features['name_mentions_room'] = 'room' in name_lower and 'bedroom' not in name_lower
    
    comfort_words = ['cosy', 'cozy', 'comfortable', 'spacious', 'bright', 'modern', 
                    'stylish', 'beautiful', 'charming', 'elegant', 'sophisticated']
    features['name_comfort_score'] = sum(1 for word in comfort_words if word in name_lower)
    
    features['name_mentions_private'] = 'private' in name_lower
    features['name_mentions_entire'] = any(word in name_lower for word in ['entire', 'whole', 'full'])
    
    view_words = ['view', 'garden', 'balcony', 'terrace', 'sea view', 'ocean view', 
                 'mountain view', 'city view', 'river view', 'park view', 'skyline']
    features['name_view_score'] = sum(1 for word in view_words if word in name_lower)
    
    features['name_mentions_central'] = any(word in name_lower for word in ['central', 'centre', 'center'])
    features['name_mentions_modern'] = any(word in name_lower for word in ['modern', 'contemporary', 'new', 'renovated'])
    
    return features

def extract_url_features(url):
    """Extract features from picture URL"""
    if pd.isna(url):
        return {
            'has_picture': False, 'url_length': 0, 'is_muscache': False,
            'image_id_length': 0, 'is_original': False, 'file_extension': 'none',
            'url_has_size_param': False, 'url_path_segments': 0,
            'estimated_image_quality': 'unknown', 'url_complexity_score': 0
        }
    
    url_str = str(url)
    features = {}
    
    features['has_picture'] = True
    features['url_length'] = len(url_str)
    features['is_muscache'] = 'muscache.com' in url_str.lower()
    
    image_id_match = re.search(r'/([a-f0-9]{8,}|[0-9]{8,})[\/_]', url_str)
    if image_id_match:
        features['image_id_length'] = len(image_id_match.group(1))
    else:
        features['image_id_length'] = 0
    
    features['is_original'] = '_original' in url_str.lower()
    
    extension_match = re.search(r'\.([a-zA-Z]{3,4})(?:\?|$)', url_str)
    if extension_match:
        features['file_extension'] = extension_match.group(1).lower()
    else:
        features['file_extension'] = 'none'
    
    features['url_has_size_param'] = any(param in url_str for param in ['im_w=', 'im_h=', 'w=', 'h='])
    features['url_path_segments'] = len([seg for seg in url_str.split('/') if seg])
    
    if '_original' in url_str:
        features['estimated_image_quality'] = 'original'
    elif any(size in url_str for size in ['_large', '_xl', '_xxl']):
        features['estimated_image_quality'] = 'large'
    elif any(size in url_str for size in ['_medium', '_med']):
        features['estimated_image_quality'] = 'medium'
    else:
        features['estimated_image_quality'] = 'standard'
    
    complexity_score = 0
    complexity_score += features['url_path_segments'] * 0.5
    complexity_score += features['image_id_length'] * 0.2
    if features['is_muscache']:
        complexity_score += 2
    if features['is_original']:
        complexity_score += 3
    features['url_complexity_score'] = complexity_score
    
    return features

# Amentities processing

def parse_amenities_simple(amenities_str):
    """Parse amenities string into list"""
    if pd.isna(amenities_str):
        return []
    
    amenities_str = str(amenities_str).strip()
    if amenities_str.startswith('[') and amenities_str.endswith(']'):
        amenities_str = amenities_str[1:-1]
    
    items = []
    for item in amenities_str.split(','):
        clean_item = item.strip().strip('"\'').strip().lower()
        if clean_item:
            items.append(clean_item)
    return items

def has_amenity_flexible(amenities_list, amenity_terms):
    """Check for amenity with flexible matching"""
    if not amenities_list:
        return False
    
    amenities_lower = [item.lower() for item in amenities_list]
    amenities_text = ' '.join(amenities_lower)
    
    for term in amenity_terms:
        term_lower = term.lower()
        if any(term_lower in amenity_lower for amenity_lower in amenities_lower):
            return True
        if term_lower in amenities_text:
            return True
    
    return False

def extract_all_amenity_features(amenities_str):
    """Extract all amenity features from amenities string"""
    amenities_list = parse_amenities_simple(amenities_str)
    features = {'amenities_count': len(amenities_list)}
    
    # Create binary features for all amenities
    for category_name, category_amenities in all_amenity_categories.items():
        for amenity, search_terms in category_amenities.items():
            col_name = f"has_{amenity}"
            features[col_name] = has_amenity_flexible(amenities_list, search_terms)
    
    # Create category counts
    for category_name, category_amenities in all_amenity_categories.items():
        category_cols = [f"has_{amenity}" for amenity in category_amenities.keys()]
        count_col = f"{category_name.lower()}_amenities_count"
        features[count_col] = sum(features.get(col, 0) for col in category_cols)
    
    # Create amenity scores
    basic_features = ['has_wifi', 'has_kitchen', 'has_tv', 'has_essentials', 'has_heating']
    features['basic_amenities_score'] = sum(features.get(col, 0) for col in basic_features)
    
    luxury_features = ['has_pool', 'has_hot_tub', 'has_gym', 'has_concierge', 'has_room_service']
    features['luxury_amenities_score'] = sum(features.get(col, 0) for col in luxury_features)
    
    convenience_features = ['has_washer', 'has_dryer', 'has_dishwasher', 'has_free_parking']
    features['convenience_amenities_score'] = sum(features.get(col, 0) for col in convenience_features)
    
    # Comfort amenities count
    comfort_features = ['has_air_conditioning', 'has_heating', 'has_fireplace', 'has_fan']
    features['comfort_amenities_count'] = sum(features.get(col, 0) for col in comfort_features)
    
    return features

# Text quality scoring

def calculate_overall_text_quality(name_features, desc_features, amenity_features):
    """Calculate comprehensive text quality score"""
    score = 0
    
    # Name contribution (25%)
    name_score = (
        name_features.get('name_luxury_score', 0) * 3 +
        name_features.get('name_location_score', 0) * 2 +
        name_features.get('name_comfort_score', 0) * 2 +
        name_features.get('name_view_score', 0) * 1.5 +
        (2 if name_features.get('name_mentions_private', False) else 0) +
        (1.5 if name_features.get('name_mentions_entire', False) else 0)
    )
    score += name_score * 0.25
    
    # Description contribution (50%)
    desc_score = (
        desc_features.get('desc_luxury_mentions', 0) * 3 +
        desc_features.get('desc_experience_mentions', 0) * 2 +
        desc_features.get('desc_cleanliness_mentions', 0) * 2.5 +
        desc_features.get('desc_safety_mentions', 0) * 2 +
        desc_features.get('desc_comfort_mentions', 0) * 2 +
        (desc_features.get('desc_sentiment_score', 0) + 1) * 5 +
        desc_features.get('desc_facility_mentions', 0) * 1.5 +
        desc_features.get('desc_location_mentions', 0) * 1.5
    )
    score += desc_score * 0.5
    
    # Amenities contribution (25%)
    amenities_score = (
        amenity_features.get('luxury_amenities_score', 0) * 4 +
        amenity_features.get('convenience_amenities_score', 0) * 2.5 +
        amenity_features.get('basic_amenities_score', 0) * 2 +
        amenity_features.get('safety_amenities_count', 0) * 2
    )
    score += amenities_score * 0.25
    
    return score

def calculate_text_intelligence_score(name_features, desc_features):
    """Calculate text intelligence score"""
    score = (
        desc_features.get('desc_readability', 0) / 10 +
        name_features.get('name_word_count', 0) * 0.5 +
        desc_features.get('desc_word_count', 0) / 50 +
        (desc_features.get('desc_sentiment_score', 0) + 1) * 2
    )
    return score

def categorize_text_appeal(text_quality_score):
    """Categorize text appeal based on quality score"""
    if text_quality_score >= 50:
        return 'Premium'
    elif text_quality_score >= 35:
        return 'High'
    elif text_quality_score >= 20:
        return 'Medium'
    elif text_quality_score >= 10:
        return 'Low'
    else:
        return 'Basic'

# Complete preprocessing

def preprocess_user_input(user_data, feature_columns, feature_defaults):
    """
    Complete preprocessing pipeline for user input
    
    Args:
        user_data: dict with user inputs
        feature_columns: list of expected feature names
        feature_defaults: dict with default values for all features
    
    Returns:
        DataFrame ready for model prediction
    """
    # Start with defaults
    processed = feature_defaults.copy()
    
    # Extract text features
    name_features = extract_name_features(user_data.get('name', ''))
    desc_features = extract_description_features(user_data.get('description', ''))
    url_features = extract_url_features(user_data.get('picture_url', ''))
    amenity_features = extract_all_amenity_features(user_data.get('amenities', ''))
    
    # Update with extracted features
    processed.update(name_features)
    processed.update(desc_features)
    processed.update(url_features)
    processed.update(amenity_features)
    
    # Basic numeric inputs
    processed['accommodates'] = user_data.get('accommodates', 2)
    processed['bedrooms'] = user_data.get('bedrooms', 1)
    processed['bathrooms'] = user_data.get('bathrooms', 1.0)
    processed['beds'] = user_data.get('beds', 1)
    processed['latitude'] = user_data.get('latitude', 53.4808)
    processed['longitude'] = user_data.get('longitude', -2.2426)
    processed['number_of_reviews'] = user_data.get('number_of_reviews', 0)
    processed['host_total_listings_count'] = user_data.get('host_total_listings_count', 1)
    
    # Review scores
    processed['review_scores_rating'] = user_data.get('review_scores_rating', 4.5)
    processed['review_scores_cleanliness'] = user_data.get('review_scores_cleanliness', 4.5)
    processed['review_scores_checkin'] = user_data.get('review_scores_checkin', 4.5)
    processed['review_scores_communication'] = user_data.get('review_scores_communication', 4.5)
    processed['review_scores_location'] = user_data.get('review_scores_location', 4.5)
    processed['review_scores_accuracy'] = user_data.get('review_scores_accuracy', 4.5)
    processed['review_scores_value'] = user_data.get('review_scores_value', 4.5)
    
    # Host information
    processed['host_is_superhost'] = 1 if user_data.get('host_is_superhost', False) else 0
    processed['host_identity_verified'] = 1 if user_data.get('host_identity_verified', False) else 0
    
    # Host days active
    if 'host_since' in user_data:
        host_since = pd.to_datetime(user_data['host_since'])
        reference_date = pd.Timestamp('2024-01-01')
        processed['host_days_active'] = max(0, (reference_date - host_since).days)
    
    # Room type one-hot encoding
    room_types = ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room']
    for rt in room_types:
        key = f'room_type_{rt}'
        if key in processed:
            processed[key] = 1 if user_data.get('room_type') == rt else 0
    
    # Property type one-hot encoding  
    property_types = ['Entire home', 'Entire condo', 'Private room', 'Entire rental unit',
                     'Entire serviced apartment', 'Entire townhouse', 'Private room in home',
                     'Private room in townhouse', 'Private room in condo', 
                     'Private room in rental unit', 'Entire cottage',
                     'Private room in bed and breakfast', 'Room in hotel']
    for pt in property_types:
        key = f'property_type_{pt}'
        if key in processed:
            processed[key] = 1 if user_data.get('property_type') == pt else 0
    
    # Neighbourhood one-hot encoding
    neighbourhoods = [
        'City Centre', 'Trafford District', 'Bury District', 'Bolton District',
        'Salford District', 'Stockport District', 'Tameside District',
        'Rochdale District', 'Oldham District', 'Wigan District',
        'Harpurhey', 'Longsight', 'Hulme', 'Old Moat', 'Fallowfield',
        'Whalley Range', 'Levenshulme', 'Didsbury West', 'Crumpsall',
        'Moss Side', 'Bradford', 'Miles Platting and Newton Heath',
        'Rusholme', 'Withington', 'Gorton South', 'Chorlton Park',
        'Chorlton', 'Cheetham', 'Ardwick', 'Gorton North',
        'Northenden', 'Woodhouse Park', 'Didsbury East'
    ]
    
    for neighbourhood in neighbourhoods:
        key = f'neighbourhood_cleansed_{neighbourhood}'
        if key in processed:
            processed[key] = 1 if user_data.get('neighbourhood_cleansed') == neighbourhood else 0
    
    # Manchester always 1
    if 'neighbourhood_group_cleansed_Manchester' in processed:
        processed['neighbourhood_group_cleansed_Manchester'] = 1
    
    # Host response time encoding
    response_times = ['within an hour', 'within a few hours', 'within a day', 'a few days or more']
    for rt in response_times:
        key = f'host_response_time_{rt}'
        if key in processed:
            processed[key] = 1 if user_data.get('host_response_time') == rt else 0
    
    # Instant bookable
    if 'instant_bookable' in processed:
        processed['instant_bookable'] = 1 if user_data.get('instant_bookable', False) else 0
    
    # Host has profile pic
    if 'host_has_profile_pic' in processed:
        processed['host_has_profile_pic'] = 1
    
    # Derived features
    accommodates = processed['accommodates']
    bedrooms = processed['bedrooms']
    
    if 'price_per_person' in processed:
        # Use median from similar listings (stored in defaults)
        processed['price_per_person'] = feature_defaults.get('price_per_person', 30)
    
    if 'people_per_bedroom' in processed:
        processed['people_per_bedroom'] = accommodates / max(bedrooms, 1)
    
    # Average review score
    if 'avg_review_score' in processed:
        review_cols = ['review_scores_rating', 'review_scores_cleanliness', 
                      'review_scores_checkin', 'review_scores_communication',
                      'review_scores_location', 'review_scores_accuracy', 'review_scores_value']
        review_values = [processed.get(col, 4.5) for col in review_cols]
        processed['avg_review_score'] = np.mean(review_values)
    
    # Calculate text quality scores
    text_quality = calculate_overall_text_quality(name_features, desc_features, amenity_features)
    if 'overall_text_quality' in processed:
        processed['overall_text_quality'] = text_quality
    
    if 'text_quality_percentile' in processed:
        # Estimate percentile based on score
        processed['text_quality_percentile'] = min(100, text_quality * 2)
    
    text_appeal = categorize_text_appeal(text_quality)
    if 'text_quality_category_Low' in processed:
        processed['text_quality_category_Low'] = 1 if text_appeal == 'Low' else 0
    if 'text_quality_category_Medium' in processed:
        processed['text_quality_category_Medium'] = 1 if text_appeal == 'Medium' else 0
    if 'text_quality_category_High' in processed:
        processed['text_quality_category_High'] = 1 if text_appeal == 'High' else 0
    if 'text_quality_category_Premium' in processed:
        processed['text_quality_category_Premium'] = 1 if text_appeal == 'Premium' else 0
    
    # Text intelligence score
    if 'text_intelligence_score' in processed:
        processed['text_intelligence_score'] = calculate_text_intelligence_score(name_features, desc_features)
    
    # Text appeal category
    if 'text_appeal_category_Low' in processed:
        processed['text_appeal_category_Low'] = 1 if text_appeal == 'Low' else 0
    if 'text_appeal_category_Medium' in processed:
        processed['text_appeal_category_Medium'] = 1 if text_appeal == 'Medium' else 0
    if 'text_appeal_category_High' in processed:
        processed['text_appeal_category_High'] = 1 if text_appeal == 'High' else 0
    if 'text_appeal_category_Premium' in processed:
        processed['text_appeal_category_Premium'] = 1 if text_appeal == 'Premium' else 0
    
    # Availability rates
    if 'availability_rate_365' in processed:
        processed['availability_rate_365'] = feature_defaults.get('availability_rate_365', 0.5)
    if 'availability_rate_30' in processed:
        processed['availability_rate_30'] = feature_defaults.get('availability_rate_30', 0.5)
    
    # Days since last review
    if 'days_since_last_review' in processed:
        processed['days_since_last_review'] = feature_defaults.get('days_since_last_review', 30)
    
    # Reviews per month
    if 'reviews_per_month' in processed:
        if processed.get('host_days_active', 0) > 0:
            months_active = processed['host_days_active'] / 30.44
            processed['reviews_per_month'] = processed['number_of_reviews'] / max(months_active, 1)
        else:
            processed['reviews_per_month'] = feature_defaults.get('reviews_per_month', 0.5)
    
    # Host acceptance/response rate
    if 'host_acceptance_rate' in processed:
        processed['host_acceptance_rate'] = feature_defaults.get('host_acceptance_rate', 90)
    if 'host_response_rate' in processed:
        processed['host_response_rate'] = feature_defaults.get('host_response_rate', 95)
    
    # Calculated host listings counts
    if 'calculated_host_listings_count' in processed:
        processed['calculated_host_listings_count'] = processed.get('host_total_listings_count', 1)
    if 'calculated_host_listings_count_private_rooms' in processed:
        if user_data.get('room_type') == 'Private room':
            processed['calculated_host_listings_count_private_rooms'] = processed.get('host_total_listings_count', 1)
        else:
            processed['calculated_host_listings_count_private_rooms'] = 0
    if 'calculated_host_listings_count_shared_rooms' in processed:
        if user_data.get('room_type') == 'Shared room':
            processed['calculated_host_listings_count_shared_rooms'] = processed.get('host_total_listings_count', 1)
        else:
            processed['calculated_host_listings_count_shared_rooms'] = 0
    
    # Create DataFrame with all features in correct order
    df = pd.DataFrame([processed])
    
    # Ensure all expected columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Return in correct order
    return df[feature_columns]
