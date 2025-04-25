import requests
import json

from dataclasses import dataclass

@dataclass
class Account:
    username: str
    password: str
    nickname: str
    userPic: str
    androidId: str
    ip: str
    token: str

    @classmethod
    def from_dict(cls, data):
        return cls(
            username=data.get('username', None),
            password=data.get('password', None),
            nickname=data.get('nickname', None),
            userPic=data.get('userPic', None),
            androidId=data.get('androidId', None),
            ip=data.get('ip', None),
            token=data.get('token', None)
        )
    
    def to_dict(cls):
        return {
            "username": cls.username,
            "password": cls.password,
            "nickname": cls.nickname,
            "userPic": cls.userPic,
            "androidId": cls.androidId,
            "ip": cls.ip,
            "token": cls.token
        }

default_headers = {
    "Host": "www.xionger.icu:8080",
    "Connection": "Keep-Alive",
    "Accept-Encoding": "gzip",
    "User-Agent": "okhttp/4.9.3",
    "Content-Type": "application/x-www-form-urlencoded"
}

class Zhenshi:
    def __init__(self, account: Account, proxy=None):
        self.account = account
        self.default_headers = default_headers.copy()
        self.proxy = proxy
        self.fakeip = account.ip
        self.Host = f"http://{default_headers['Host']}"
        self.default_headers["X-Forwarded-For"] = self.fakeip
        if account.token is None:
            self.login()
        else:
            self.token = account.token
        self.default_headers["Authorization"] = account.token

    
    def send_request(self, endpoint, headers, data = None, method='POST'):
        submit_headers = headers.copy()  # Create a copy to avoid shared state issues
        # submit_headers["Content-Length"] = str(len(data))
        try:
            # Print debug info
            # print(f"Sending request to {self.Host}/{endpoint} with headers: {submit_headers} and data: {data}")
            response = requests.request(method=method, url=f"{self.Host}/{endpoint}", headers=submit_headers, data=data, proxies=self.proxy)
            response.raise_for_status()  # Raise an error for bad responses
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        
    def login(self):
        endpoint = "user/login"
        data = {
            "username": self.account.username,
            "password": self.account.password,
            "nickname": self.account.nickname,
            "userPic": self.account.userPic,
            "androidId": self.account.androidId
        }
        response = self.send_request(endpoint, self.default_headers, data)
        if response and response['code'] == 200:
            self.account.token = response['message']
            self.token = response['message']
            self.default_headers["Authorization"] = self.token
            return response
        else:
            print(response)
            raise ValueError("Login failed or invalid response")
        
    def check(self):
        endpoint = "user/checked"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response
        
    def get_userinfo(self):
        endpoint = "user/userInfo"
        data = ""
        response = self.send_request(endpoint, self.default_headers, data)
        return response

    
    def get_kline(self, code):
        endpoint = "stock/getKLineByCode"
        data = {"code": code}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_orderbook(self, code):
        endpoint = "trade/priceList"
        data = {"code": code}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userstock(self, code):
        endpoint = "stock/getUserStockByCode"
        data = {"code": code}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_allorder(self):
        endpoint = "trade/getAllOrder"
        response = self.send_request(endpoint, self.default_headers)
        return response
    
    def addorder(self, code, number, price, operation):
        endpoint = "trade/addOrder"
        data = {"code": code, "number": number, "price": price, "operation": operation}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def cancelorder(self, id):
        endpoint = "trade/cancelOrder"
        data = {"id": id}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def add_lotteryorder(self, number, multiple):
        endpoint = "lottery/addLotteryOrder"
        data = {"number": number, "multiple": multiple}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_userlotteryorder(self):
        endpoint = "lottery/getUserLotteryOrder"
        response = self.send_request(endpoint, self.default_headers)
        return response
    
    def delete_lotteryorder(self, id):
        endpoint = "lottery/deleteLotteryOrder"
        data = {"id": id}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def reward(self):
        endpoint = "user/reward"
        response = self.send_request(endpoint, self.default_headers)
        return response
    
    def update_nickname(self, nickname):
        endpoint = "user/updateNickname"
        data = {"nickname": nickname}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_rank(self):
        endpoint = "user/getRank"
        response = self.send_request(endpoint, self.default_headers)
        return response
    
    def get_rank_money(self):
        endpoint = "user/getRankMoney"
        response = self.send_request(endpoint, self.default_headers)
        return response

    def cancel_all_orders(self):
        orders = self.get_allorder()['data']
        for o in orders:
            self.cancelorder(o['id'])

    def get_other_userstock_byname(self, username):
        endpoint = "stock/getOtherUserStockByUsername"
        data = {"username": username}
        response = self.send_request(endpoint, self.default_headers, data)
        return response
    
    def get_stocklist(self):
        endpoint = "stock/list"
        response = self.send_request(endpoint, self.default_headers, method='GET')
        return response