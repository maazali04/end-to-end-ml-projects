import requests

API_URL = "https://titanic-predictor-api-two.vercel.app/predict"

payload = {
    "Name": "Braund, Mr. Owen Harris",
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.25,
    "Embarked": "S"
}
def test_prediction():
    print(f"Sending request to: {API_URL} ...\n")
    
    try:
        response = requests.post(API_URL, json=payload)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API Request Successful!")
            print("Response JSON:")
            print(response.json())
        else:
            print("❌ API returned an error:")
            print(response.text)

    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")

if __name__ == "__main__":
    test_prediction()