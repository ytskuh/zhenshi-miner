import requests
import json

default_headers = {
    "Host": "www.xionger.icu:8080",
    "Connection": "Keep-Alive",
    "Accept-Encoding": "gzip",
    "User-Agent": "okhttp/4.9.3",
    "Content-Type": "application/x-www-form-urlencoded"
}

class Zhenshi:
    def __init__(self, account, ):
        self.Host = "http://www.xionger.icu:8080"
        self.default_headers = default_headers
        self.default_headers["Authorization"] = account
    
    def send_request(self, endpoint, headers, data):
        submit_headers = headers.copy()  # Create a copy to avoid shared state issues
        submit_headers["Content-Length"] = str(len(data))
        try:
            response = requests.post(f"{self.Host}/{endpoint}", headers=submit_headers, data=data)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        
    def get_userinfo(self):
        endpoint = "user/userInfo"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response

    
    def get_kline(self, code):
        endpoint = "stock/getKLineByCode"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_orderbook(self, code):
        endpoint = "trade/priceList"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userstock(self, code):
        endpoint = "stock/getUserStockByCode"
        data = f"code={code}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_allorder(self):
        endpoint = "trade/getAllOrder"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def addorder(self, code, number, price, operation):
        endpoint = "trade/addOrder"
        data = f"code={code}&number={number}&price={price}&operation={operation}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def add_lotteryorder(self, number, multiple):
        endpoint = "lottery/addLotteryOrder"
        data = f"number={number}&multiple={multiple}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userlotteryorder(self):
        endpoint = "lottery/getUserLotteryOrder"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def delete_lotteryorder(self, id):
        endpoint = "lottery/deleteLotteryOrder"
        data = f"id={id}"
        response = self.send_request(endpoint, self.default_headers, data)
        return response