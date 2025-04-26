import json
from dataclasses import dataclass
from typing import List
import asyncio
from collections import defaultdict
from typing import Any, Optional, Callable, Tuple
import random
import os

import utils

from zhenshiapi import Account, Zhenshi

class AccountStorage:
    """Handles reading and writing account data to a file."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def save_accounts(self, accounts: List[Account]):
        """Saves accounts to a JSON file."""
        try:
            with open(self.file_path, 'w') as f:
                json.dump([Account.to_dict(account) for account in accounts], f, indent=4)
        except IOError as e:
            raise IOError(f"Failed to save accounts: {e}")

    def load_accounts(self) -> List[Account]:
        """Loads accounts from a JSON file."""
        if not os.path.exists(self.file_path):
            return []
        try:
            with open(self.file_path, 'r') as f:
                accounts_data = json.load(f)
                return [Account.from_dict(data) for data in accounts_data]
        except (IOError, json.JSONDecodeError) as e:
            raise IOError(f"Failed to load accounts: {e}")


class BatchExecutionHandler:
    """Manages the status and results of a batch execution of operations."""
    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.results = defaultdict(list)  # Maps account username to list of (success, result/error)
        self._tasks = []
        self._running = True

    def add_task(self, task: asyncio.Task):
        """Add a task to the handler."""
        self._tasks.append(task)

    async def wait_completion(self, timeout: Optional[float] = None) -> None:
        """Wait for all tasks to complete, with an optional timeout."""
        try:
            await asyncio.wait(self._tasks, timeout=timeout)
        except asyncio.TimeoutError:
            self._running = False
            raise TimeoutError("Batch execution timed out")

    def is_running(self) -> bool:
        """Check if the batch execution is still running."""
        return self._running and self.completed_tasks < self.total_tasks

    def get_progress(self) -> float:
        """Get the progress of the batch execution as a percentage."""
        return (self.completed_tasks / self.total_tasks) * 100 if self.total_tasks > 0 else 0.0

    def get_results(self) -> dict:
        """Get the results of all completed tasks."""
        return dict(self.results)

    def record_result(self, username: str, success: bool, result: Any) -> None:
        """Record the result of a task for an account."""
        self.results[username].append((success, result))
        self.completed_tasks += 1
        if self.completed_tasks >= self.total_tasks:
            self._running = False

class Cluster:
    def __init__(self, accounts: List[Account] = None, storage_file: Optional[str] = None):
        self.accounts = accounts
        if storage_file:
            self.storage = AccountStorage(storage_file)
            self.storage.save_accounts(accounts)
        else:
            self.storage = None

    def from_json(self, storage_file):
        self.storage = AccountStorage(storage_file)
        self.accounts = self.storage.load_accounts()

    def create_account(self) -> Account:
        """Creates a new account with random values."""
        username = utils.generate_random_username()
        password = utils.username_to_password(username)
        nickname = utils.generate_random_nickname()
        userPic = utils.generate_random_userPic()
        androidId = utils.generate_random_androidId()
        ip = utils.generate_random_ip()
        
        account = Account(username, password, nickname, userPic, androidId, ip, None)
        self.accounts.append(account)
        self.storage.save_accounts(self.accounts)
        
        return account
    
    def add_account(self, account: Account) -> None:
        """Adds an existing account to the cluster."""
        self.accounts.append(account)
        self.storage.save_accounts(self.accounts)
    
    def remove_account(self, account: Account) -> None:
        """Removes an account from the cluster."""
        self.accounts.remove(account)
        self.storage.save_accounts(self.accounts)
    
    def update_account(self, account: Account) -> None:
        """Updates an existing account in the cluster."""
        for i, acc in enumerate(self.accounts):
            if acc.username == account.username:
                self.accounts[i] = account
                break
        self.storage.save_accounts(self.accounts)
    
    def get_account(self, username: str) -> Optional[Account]:
        """Retrieves an account by username."""
        for account in self.accounts:
            if account.username == username:
                return account
        return None
    
    def batch_execution(self, method: Callable, args_list: List[Tuple[Any, ...]], total_time: float) -> BatchExecutionHandler:
        """
        Execute a Zhenshi method asynchronously for multiple accounts, distributed over a time period.
        
        Args:
            method: The Zhenshi member function to execute (e.g., Zhenshi.check).
            args_list: List of argument tuples for each account's method call.
            total_time: Time (in seconds) over which to distribute the operations.
        
        Returns:
            BatchExecutionHandler: A handler to track the batch execution.
        """
        if len(args_list) != len(self.accounts):
            raise ValueError("args_list length must match the number of accounts")

        handler = BatchExecutionHandler(len(self.accounts))

        async def execute_for_account(account: Account, args: Tuple[Any, ...], delay: float) -> None:
            """Execute the method for a single account after a delay."""
            try:
                await asyncio.sleep(delay)  # Introduce delay to stagger execution
                zhenshi = Zhenshi(account)
                result = await asyncio.get_event_loop().run_in_executor(None, lambda: method(zhenshi, *args))
                handler.record_result(account.username, True, result)
                # Update account in case token or other fields changed
                self.accounts[self.accounts.index(account)] = account
                self.storage.save_accounts(self.accounts)
            except Exception as e:
                handler.record_result(account.username, False, str(e))

        async def run_batch():
            """Run the batch execution with staggered delays."""
            delays = [random.uniform(0, total_time) for _ in range(len(self.accounts))]
            delays.sort()  # Ensure delays are in ascending order for smoother distribution
            tasks = [
                asyncio.create_task(execute_for_account(account, args, delay))
                for account, args, delay in zip(self.accounts, args_list, delays)
            ]
            for task in tasks:
                handler.add_task(task)

        # Start the batch execution in the background
        asyncio.create_task(run_batch())
        return handler