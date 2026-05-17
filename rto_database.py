"""
Indian RTO Database
State codes, RTO codes, vehicle class info derived from number plate
"""

STATE_CODES = {
    "AN": "Andaman & Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman & Diu",
    "DL": "Delhi",
    "DN": "Dadra & Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu & Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TG": "Telangana",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

# Major RTO offices
RTO_CODES = {
    "MH01": "Mumbai (Central)", "MH02": "Mumbai (West)", "MH03": "Mumbai (East)",
    "MH04": "Thane", "MH05": "Kalyan", "MH06": "Raigad", "MH07": "Ratnagiri",
    "MH08": "Sindhudurg", "MH09": "Kolhapur", "MH10": "Sangli", "MH11": "Satara",
    "MH12": "Pune", "MH13": "Solapur", "MH14": "Ahmednagar", "MH15": "Nashik",
    "MH16": "Dhule", "MH17": "Jalgaon", "MH18": "Aurangabad", "MH19": "Bid",
    "MH20": "Osmanabad", "MH21": "Nanded", "MH22": "Latur", "MH23": "Parbhani",
    "MH24": "Yavatmal", "MH25": "Amravati", "MH26": "Akola", "MH27": "Buldhana",
    "MH28": "Washim", "MH29": "Nagpur", "MH30": "Wardha", "MH31": "Bhandara",
    "MH32": "Gondia", "MH33": "Chandrapur", "MH34": "Gadchiroli", "MH35": "Nandurbar",
    "MH36": "Hingoli", "MH37": "Jalna", "MH38": "Palghar", "MH43": "Mumbai (South)",
    "DL01": "New Delhi (North)", "DL02": "Delhi (North East)", "DL03": "Delhi (South West)",
    "DL04": "Delhi (South)", "DL05": "Delhi (East)", "DL06": "Delhi (West)",
    "DL07": "Delhi (North West)", "DL08": "Delhi (Shahdara)", "DL09": "Delhi (North West 2)",
    "DL10": "Delhi (South East)", "DL11": "Delhi (Central)", "DL12": "Delhi (Rohini)",
    "KA01": "Bangalore Central", "KA02": "Bangalore North", "KA03": "Bangalore East",
    "KA04": "Bangalore South", "KA05": "Belgaum", "KA09": "Davanagere",
    "KA19": "Mysore", "KA41": "Bangalore (New)", "KA50": "Bangalore (Latest)",
    "TN01": "Chennai (Central)", "TN02": "Chennai (North)", "TN03": "Chennai (South)",
    "TN09": "Coimbatore", "TN11": "Madurai", "TN18": "Tiruchirappalli",
    "GJ01": "Ahmedabad (Central)", "GJ02": "Surat", "GJ03": "Vadodara", "GJ04": "Rajkot",
    "RJ01": "Jaipur", "RJ02": "Jodhpur", "RJ14": "Ajmer", "RJ19": "Udaipur",
    "UP01": "Lucknow", "UP14": "Agra", "UP15": "Kanpur", "UP16": "Ghaziabad",
    "UP32": "Noida", "UP65": "Prayagraj", "UP80": "Varanasi",
    "WB01": "Kolkata", "WB02": "Howrah", "WB06": "Durgapur",
    "AP01": "Hyderabad", "AP09": "Vishakhapatnam", "AP11": "Vijaywada",
    "HR01": "Ambala", "HR26": "Gurugram", "HR55": "Faridabad",
    "PB01": "Chandigarh", "PB10": "Ludhiana",
    "MP04": "Bhopal", "MP09": "Indore",
    "BR01": "Patna",
    "OR01": "Bhubaneswar", "OD01": "Bhubaneswar",
    "AS01": "Guwahati",
    "JH01": "Ranchi",
    "CG01": "Raipur",
    "HP01": "Shimla",
    "KL01": "Thiruvananthapuram", "KL07": "Kozhikode", "KL09": "Kochi",
    "GA01": "Panaji", "GA02": "Margao",
    "TG01": "Hyderabad", "TS01": "Hyderabad",
}

# Vehicle class indicators from series letter
VEHICLE_CLASS_HINTS = {
    "A": "Private Car / Auto",
    "B": "Bus / Private Vehicle",
    "C": "Commercial / Cab",
    "D": "Defence / Government",
    "E": "Electric Vehicle",
    "F": "Farm / Agricultural",
    "G": "Government Vehicle",
    "H": "Hire / Taxi",
    "P": "Private",
    "T": "Taxi / Transport",
    "V": "Van / Goods",
}

# Fuel type (modern BH series)
BH_SERIES_INFO = {
    "0E": "Electric",
    "0D": "Diesel",
    "0P": "Petrol",
    "0H": "Hybrid",
    "0G": "Gas (CNG/LPG)",
}


def parse_indian_number_plate(plate_text: str) -> dict:
    """
    Parse Indian number plate and return vehicle information.
    
    Formats supported:
    - Old: MH 12 AB 1234   (State + RTO + Series + Number)
    - BH Series: 22 BH 1234 A (Year + BH + Number + Fuel)
    - Electric: MH 12 EA 1234 (same format but EA/EB etc series for electric)
    """
    import re

    plate = plate_text.upper().replace(" ", "").replace("-", "").strip()
    result = {
        "raw_text": plate_text.upper().strip(),
        "cleaned_plate": plate,
        "plate_type": "Unknown",
        "state": "Unknown",
        "rto_office": "Unknown",
        "series": "Unknown",
        "vehicle_number": "Unknown",
        "vehicle_class": "Unknown",
        "is_electric": False,
        "is_bh_series": False,
        "registration_year": "Unknown",
        "fuel_type": "Unknown",
        "valid": False,
        "confidence": "Low",
    }

    if len(plate) < 6:
        result["error"] = "Plate text too short"
        return result

    # --- BH Series: e.g. 22BH1234A ---
    bh_match = re.match(r'^(\d{2})BH(\d{4})([A-Z]{1,2})$', plate)
    if bh_match:
        year_code = bh_match.group(1)
        number = bh_match.group(2)
        fuel_code = bh_match.group(3)
        result["plate_type"] = "BH Series (Bharat)"
        result["registration_year"] = f"20{year_code}"
        result["vehicle_number"] = number
        result["is_bh_series"] = True
        result["state"] = "Pan-India (BH Series)"
        result["rto_office"] = "Pan-India Registration"
        fuel_key = f"0{fuel_code[0]}" if len(fuel_code) == 1 else fuel_code
        result["fuel_type"] = BH_SERIES_INFO.get(fuel_key, f"Code: {fuel_code}")
        if fuel_code == "E":
            result["is_electric"] = True
            result["fuel_type"] = "Electric"
        result["valid"] = True
        result["confidence"] = "High"
        return result

    # --- Standard: e.g. MH12AB1234 ---
    std_match = re.match(r'^([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{1,4})$', plate)
    if std_match:
        state_code = std_match.group(1)
        rto_num = std_match.group(2)
        series = std_match.group(3)
        number = std_match.group(4)

        rto_key = f"{state_code}{rto_num}"

        result["plate_type"] = "Standard Indian"
        result["state"] = STATE_CODES.get(state_code, f"Unknown ({state_code})")
        result["rto_office"] = RTO_CODES.get(rto_key, f"RTO {rto_key}")
        result["series"] = series
        result["vehicle_number"] = number.zfill(4)
        result["valid"] = True
        result["confidence"] = "High"

        # Electric vehicle detection
        if series.startswith("E") and state_code in ("MH", "DL", "KA", "TN", "GJ"):
            result["is_electric"] = True
            result["fuel_type"] = "Electric (likely)"
            result["vehicle_class"] = "Electric Vehicle"
        else:
            first_letter = series[0]
            result["vehicle_class"] = VEHICLE_CLASS_HINTS.get(first_letter, "Private / General")

        return result

    # --- Partial match attempt ---
    partial = re.match(r'^([A-Z]{2})(\d{2})', plate)
    if partial:
        state_code = partial.group(1)
        rto_num = partial.group(2)
        rto_key = f"{state_code}{rto_num}"
        result["plate_type"] = "Partial Read"
        result["state"] = STATE_CODES.get(state_code, f"Unknown ({state_code})")
        result["rto_office"] = RTO_CODES.get(rto_key, f"RTO {rto_key}")
        result["confidence"] = "Medium"
        result["valid"] = True
        return result

    result["error"] = "Could not parse plate format"
    return result
