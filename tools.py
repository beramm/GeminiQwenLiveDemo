
import os 
import logging
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(
    level=logging.INFO,     
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(filename)s:%(lineno)d | "
        "%(funcName)s() | "
        "%(message)s () | "
    )
)
logger = logging.getLogger(__name__)

SERVICE_NAME = "itinerary-service"
ENV = os.getenv("ENV", "dev")


url = "https://places.googleapis.com/v1/places:searchText"

headers_places = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": os.getenv("GMAP_API_KEY"),
    "X-Goog-FieldMask": "places.id,places.formattedAddress,places.displayName,places.location,places.types,places.rating,places.userRatingCount,places.priceRange,places.regularOpeningHours,places.photos"
}

headers_hotels = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": os.getenv("GMAP_API_KEY"),
    "X-Goog-FieldMask": "places.id,places.formattedAddress,places.displayName,places.location,places.types,places.rating,places.userRatingCount,places.priceRange,places.regularOpeningHours,places.photos"
}

LODGING_TYPES = {
    "lodging", 
    "hotel", 
    "motel", 
    "extended_stay_hotel", 
    "resort_hotel", 
    "bed_and_breakfast", 
    "hostel",
    "budget_japanese_inn",
    "campground",
    "camping_cabin",
    "cottage", 
    "farmstay", 
    "guest_house", 
    "inn", 
    "japanese_inn", 
    "mobile_home_park", 
    "private_guest_room", 
    "rv_park"}

def search_places(query):
    data = {
        "textQuery": query,
        "maxResultCount": 3
    }

    response = requests.post(url, headers=headers_places, json=data)

    if response.status_code == 200:
        result = response.json()
        places = result.get("places", [])
        result["places"] = [p for p in places if not LODGING_TYPES.intersection(p.get("types", []))]
        return result
    else:
        logger.error(f"Error searching places: {response.status_code} - {response.text}")
        return None
    
def search_hotels(query):
    data = {
        "textQuery": query,
        "includedType": "lodging",
        "maxResultCount": 3
    }

    response = requests.post(url, headers=headers_hotels, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Error searching hotels: {response.status_code} - {response.text}")
        return None
    
