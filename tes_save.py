import json


def save_to_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)
from datetime import datetime
now = datetime.now()
formatted_date = now.strftime("%d %B %Y, %H:%M")
def write_json(new_data, filename):
    with open(filename, 'r+') as file:
        file_data = json.load(file)
        file_data["results"].append({
            "data": new_data,
            "date": formatted_date
        })
        
        file.seek(0)
        json.dump(file_data, file, indent=4)
        file.truncate()
data = {
    "head": 20,
    "chest": 12,
    "abdomen": 31,
    "hand": 21,
    "leg": 11,
    "height": 30
}
# data = 20,12,31,21,112    
# save_to_json(data, "results.json")
write_json(data, "results.json")

def load_list_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

data_load = load_list_from_json('results.json')
print(data_load)
            
            