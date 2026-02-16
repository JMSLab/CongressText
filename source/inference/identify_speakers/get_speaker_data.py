import requests, yaml
import pandas as pd

current_URL = "https://unitedstates.github.io/congress-legislators/legislators-current.yaml"
historical_URL = "https://unitedstates.github.io/congress-legislators/legislators-historical.yaml"

def read_yaml(url):
    response = requests.get(url, timeout=30)
    text = response.content.decode("utf-8", errors="replace")
    data = yaml.safe_load(text)
    return data

# Get current legislators
current_data = read_yaml(current_URL)
# Get historical legislators
historical_data = read_yaml(historical_URL)

# Convert YAML to pandas df
# Each row is a legislator - session of Congress
# extract firstname, lastname, nickname, chamber, congress, icpsr, district_code, state_abbrev
# also collect leadership roles

def process_legislators(data):
    rows = []
    
    for legislator in data:
        # Extract basic info
        icpsr = legislator.get('id', {}).get('icpsr')
        name = legislator.get('name', {})
        first_name = name.get('first')
        last_name = name.get('last')
        nickname = name.get('nickname')
        gender = legislator.get('bio', {}).get('gender')
        
        # Process leadership roles
        leadership_roles = legislator.get('leadership_roles', [])
        
        # Process each term
        terms = legislator.get('terms', [])
        for term in terms:
            # Calculate congress number range from start and end dates
            start_date = term.get('start')
            end_date = term.get('end')
            if start_date:
                start_year = int(start_date.split('-')[0])
                start_congress = ((start_year - 1789) // 2) + 1
                
                if end_date:
                    end_year = int(end_date.split('-')[0])
                    if end_year > start_year:
                        # don't +1 because goes into next Congress
                        end_congress = ((end_year - 1789) // 2) 
                    else:
                        end_congress = start_congress
                else:
                    # If no end date, assume current
                    end_congress = start_congress
            else:
                continue  # Skip if no start date
                
            # Determine chamber
            term_type = term.get('type')
            if term_type == 'rep':
                chamber = 'house'
            elif term_type == 'sen':
                chamber = 'senate'
            else:
                chamber = term_type
                
            # Get state and district
            state_abbrev = term.get('state')
            district = term.get('district')
            district_code = f"{state_abbrev}{district:02d}" if district and state_abbrev else state_abbrev
            
            # Create a row for each congress in the term range
            for congress in range(start_congress, end_congress + 1):
                # Calculate the start and end dates for this specific congress
                congress_start_year = 1789 + (congress - 1) * 2
                congress_end_year = congress_start_year + 1
                congress_start_date = f"{congress_start_year}-01-03"
                congress_end_date = f"{congress_end_year}-12-31"
                
                # Find leadership roles that overlap with this specific congress
                term_leadership_roles = []
                for role in leadership_roles:
                    role_start = role.get('start')
                    role_end = role.get('end')
                    role_chamber = role.get('chamber')
                    
                    # Check if role overlaps with this congress and matches chamber
                    if (role_chamber == chamber or role_chamber is None) and role_start:
                        # Check overlap with this specific congress
                        if role_start <= congress_end_date and (role_end or '9999-12-31') >= congress_start_date:
                            term_leadership_roles.append(role.get('title', ''))
                
                # Join leadership roles with "/"
                leadership_str = '/'.join(term_leadership_roles) if term_leadership_roles else ''
                
                # Create row for this congress
                row = {
                    'first_name': first_name,
                    'last_name': last_name,
                    'nickname': nickname,
                    'chamber': chamber,
                    'congress': congress,
                    'icpsr': icpsr,
                    'district_code': district_code,
                    'state_abbrev': state_abbrev,
                    'leadership_roles': leadership_str,
                    'gender': gender
                }
                rows.append(row)
    
    return pd.DataFrame(rows)

# Process current and historical data
current_csv = process_legislators(current_data)
historical_csv = process_legislators(historical_data)

# Combine current and historical legislators (append)
all_csv = pd.concat([current_csv, historical_csv])

# Save to CSV
all_csv.to_csv("datastore/inference/congress_legislators.csv", index=False)
all_csv.to_csv("datastore/inference/daily_harmonized/congress_legislators.csv", index=False)
