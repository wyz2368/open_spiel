import numpy as np
import copy
import os
import itertools
from collections import OrderedDict
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
# from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

# a = set()
# a.add((1,2,3,1,2,4,4,5))
# a.add((1,2,3))
# a.add((1,2,3,1,2,4,4,5))
#
# print(a)

class Solution:
    """
    @param nums: An integer array sorted in ascending order
    @param target: An integer
    @return: An integer
    """

    def findPosition(self, nums, target):
        # write your code here
        if len(nums) == 0 or not nums:
            return -1

        start, end = 0, len(nums) - 1
        print("start, end:", start, end)
        while start + 1 < end:
            mid = int(start + (end - start) / 2)
            if nums[mid] == target:
                return mid
            if nums[mid] > target:
                start = mid + 1
            if nums[mid] < target:
                end = mid - 1

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end

        return -1

nums = [1,2,2,4,5,5]
target = 5

S = Solution()
print(S.findPosition(nums, target))