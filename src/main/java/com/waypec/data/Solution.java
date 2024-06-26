package com.waypec.data;

import java.util.*;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            } else {
                map.put(nums[i], i);
            }
        }
        return new int[]{-1, -1};
    }

    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (int i = 0; i < strs.length; i++) {
            char[] array = strs[i].toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<>());
            list.add(strs[i]);
            map.put(key, list);
        }
        return new ArrayList<>(map.values());
    }

    //739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int[] res = new int[temperatures.length];
        for (int i = 1; i < temperatures.length; i++) {
            if (temperatures[i] <= temperatures[stack.peek()]) {
                stack.push(i);
            } else {
                while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                    res[stack.peek()] = i - stack.peek();
                    stack.pop();
                }
                stack.push(i);
            }
        }
        return res;
    }

    //496.下一个更大元素 I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] res = new int[nums1.length];
        Arrays.fill(res, -1);
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; i++) {
            map.put(nums1[i], i);
        }
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        for (int i = 1; i < nums2.length; i++) {
            if (nums2[i] <= nums2[stack.peek()]) {
                stack.push(i);
            } else {
                while (!stack.isEmpty() && nums2[i] > nums2[stack.peek()]) {
                    if (map.containsKey(nums2[stack.peek()])) {
                        int index = map.get(nums2[stack.peek()]);
                        res[index] = nums2[i];
                    }
                    stack.pop();
                }
                stack.push(i);
            }
        }
        return res;
    }

    //503.下一个更大元素II 循环数组
    public int[] nextGreaterElements(int[] nums) {

        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        //循环数组的处理方法是模拟遍历两遍nums数组
        for (int i = 1; i < nums.length * 2; i++) {
            //index是遍历的元素下标，
            //peek是待比较元素的下标
            int index = i % nums.length;
            if (nums[index] <= nums[stack.peek()]) {
                stack.push(index);
            } else {
                //拿遍历的元素和待比较的元素一直比较，并不断更新待比较的元素，注意遍历条件
                while (!stack.isEmpty() && nums[index] > nums[stack.peek()]) {
                    res[stack.peek()] = nums[index];
                    //注意pop之后不应该再去peek，以防止出现pop之后栈为空，peek不出来的情况
                    stack.pop();
                }
                stack.push(index);
            }
        }
        return res;
    }

    //42. 接雨水
    //解题思路1：找高个中的矮个，找该列元素左边最大值、右边最大值中的较小值，注意第一个和最后一个不需要计算
    //解题思路2：单调栈
    public int trap(int[] height) {
        //第一个和最后一个柱子不需要计算，默认值为0
        int[] maxLeft = new int[height.length];
        int[] rightMax = new int[height.length];
        int sum = 0;
        int left = height[0];
        for (int i = 1; i < height.length - 1; i++) {
            maxLeft[i] = left;
            left = Math.max(left, height[i]);
        }
        return sum;
    }

    //84.柱状图中最大的矩形
    //栈顶和栈顶的下一个元素以及要入栈的三个元素组成了我们要求最大面积的高度和宽度
    public int largestRectangleArea(int[] heights) {
        int[] newHeights = new int[heights.length + 2];
        newHeights[0] = 0;
        newHeights[newHeights.length - 1] = 0;
        for (int i = 1; i < newHeights.length - 1; i++) {
            newHeights[i] = heights[i - 1];
        }

        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int res = 0;
        for (int i = 1; i < newHeights.length; i++) {
            //遍历元素的高度大于等于栈顶下标元素的高度
            if (newHeights[i] >= newHeights[stack.peek()]) {
                stack.push(i);
            } else {
                while (!stack.isEmpty() && newHeights[i] < newHeights[stack.peek()]) {
                    int mid = stack.peek();
                    stack.pop();
                    if (!stack.isEmpty()) {
                        int w = i - stack.peek() - 1;
                        int h = newHeights[mid];
                        res = Math.max(res, w * h);
                    }

                }
                stack.push(i);
            }
        }
        return res;
    }


    //20. 有效的括号
    public boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '(') {
                stack.push(')');
            } else if (c == '[') {
                stack.push(']');
            } else if (c == '{') {
                stack.push('}');
            } else if (stack.isEmpty() || c != stack.peek()) {
                return false;
            } else {
                stack.pop();
            }
        }
        return stack.isEmpty();
    }

    public int fib(int n) {
        if (n < 2) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int climbStairs(int n) {
        if (n < 3) {
            return n;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];

    }

    //62.不同路径
    public int uniquePaths(int m, int n) {
        //dp[i][j] 表示从坐标[0][0]到坐标[i][j]共有dp[i][j]种不同的路径可以到达
        int[][] dp = new int[m][n];

        //初始化二维数组的第一列和第一行
        for (int i = 0; i < m; i++) {
            dp[i][0] = 1;
        }
        for (int j = 0; j < n; j++) {
            dp[0][j] = 1;
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }


    //63. 不同路径 II
    //相比较于上一题添加了障碍物，在初始化，和推导dp公式的时候需要考虑进去
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] dp = new int[m][n];
        //初始化，需要考虑障碍物,一旦到了障碍物，后面的到达路径就为0，不需要计算
        for (int i = 0; i < m; i++) {
            if (obstacleGrid[i][0] == 0) {
                dp[i][0] = 1;
            } else {
                break;
            }
        }


        for (int j = 0; j < n; j++) {
            if (obstacleGrid[0][j] == 0) {
                dp[0][j] = 1;
            } else {
                break;
            }
        }

        //dp公式递推的时候也需要考虑障碍物，有障碍物的格子，不需要计算
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if (obstacleGrid[i][j] == 0) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    //整数拆分
    public int integerBreak(int n) {
        //dp[i]表示拆分整数i得到的最大乘积
        int[] dp = new int[n + 1];
        dp[2] = 1;

        for (int i = 3; i <= n; i++) {
            for (int j = 2; j < i; j++) {
                //dp[j] * (i-j)表示拆分j与(i-j)的乘积，理解为多个拆分,j从2开始才有意义
                //j*(i-j)理解为两个拆分
                //对于某个i来说，j需要遍历直到i-1,取两个当中最大值即为dp[i]
                dp[i] = Math.max(dp[i], Math.max(dp[j] * (i - j), j * (i - j)));
            }
        }
        return dp[n];
    }

    //不同的二叉搜索树
    /*
    * 首先定义一个函数G[n]:表示1...n构成的二叉搜索树的个数。1...n序列中的每个点都可以当做根节点，
    * 该节点左边的序列构成左子树，右边的序列构成右子树。比如给定一个序列[1,2,3,4,5,6,7],我们选取节点3为根结点。
    * 左子树为[1,2]构成，可以使用G[2]来表示；右子树[4,5,6,7]可以使用G[4]来表示
    * （由G[n]的定义可知，[4,5,6,7]和[1,2,3,4]构成的子树个数相同）。
    * 我们使用f(3, 7)来表示序列长度为7，根结点为3时构成的二叉搜索树的个数，
    * 则f(3,7) = G[2] * G[4];我们可以推导出f(i, n) = G[i-1] * G[n - i].
    由以上分析可知：
    G[n] = G[0] * G[n-1] + G[1] * G[n - 2].....G[n-1] * G[0];
    分别选取每一个数作为根结点。由上式可知，要计算出G[n]，需要先计算出G[0],G[1]...，运用动态规划的思想求解。
    * */
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            //对于一个固定的i，取j为根节点
            for (int j = 1; j <= i; j++) {
                //以j为根节点，dp[j-1]为左子树1,j-1构成二叉搜索树的个数，dp[i-j]为右子树构成二叉搜索树的个数
                dp[i] += dp[j - 1] * dp[i - j];
            }
        }
        return dp[n];
    }

    //背包理论： https://www.programmercarl.com/%E8%83%8C%E5%8C%85%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%8001%E8%83%8C%E5%8C%85-2.html#%E6%80%9D%E8%B7%AF
    public int weightValueBag() {
        int[] weight = new int[]{1, 3, 4};
        int[] value = new int[]{15, 20, 30};

        int bagWeight = 4;
        int[] dp = new int[bagWeight + 1];
        Arrays.fill(dp, 0);
        //注意顺序，先遍历物品，再遍历重量
        for (int i = 0; i < weight.length; i++) {
            //如果物品重量>背包重量，则没必要放入
            for (int j = bagWeight; j >= weight[i]; j++) {
                dp[j] = Math.max(dp[i], dp[j - weight[i]] + value[i]);
            }
        }
        return dp[bagWeight];
    }

    //416. 分割等和子集
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if (sum % 2 != 0) {
            return false;
        }
        int dp[] = new int[sum + 1];
        int target = sum / 2;
        for (int i = 0; i < nums.length; i++) {
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - nums[i]] + nums[i]);
            }
        }
        return dp[target] == target;
    }


    //    1049.最后一块石头的重量II
    public int lastStoneWeightII(int[] stones) {
        int sum = 0;
        for (int i = 0; i < stones.length; i++) {
            sum += stones[i];
        }
        int target = sum / 2;
        int[] dp = new int[target + 1];
        //dp[j]表示0-i选物品，容量为j的背包，最大可以装dp[j]的价值
        for (int i = 0; i < stones.length; i++) {
            for (int j = target; j >= stones[i]; j--) {
                dp[j] = Math.max(dp[j], dp[j - stones[i]] + stones[i]);
            }
        }
        return sum - 2 * dp[target];
    }

    //494.目标和
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
        }
        if ((target + sum) % 2 == 1) {
            return 0;
        }
        if (Math.abs(target) > sum) {
            return 0;
        }
        int x = (target + sum) / 2;
        int[] dp = new int[x + 1];
        dp[0] = 1;
        //dp[j]表示从[0-i]取物品，最大价值为j的种类有dp[j]种

        //例如：dp[j]，j 为5，
        //已经有一个1（nums[i]） 的话，有 dp[4]种方法 凑成 容量为5的背包。
        //已经有一个2（nums[i]） 的话，有 dp[3]种方法 凑成 容量为5的背包。
        //已经有一个3（nums[i]） 的话，有 dp[2]中方法 凑成 容量为5的背包
        //已经有一个4（nums[i]） 的话，有 dp[1]中方法 凑成 容量为5的背包
        //已经有一个5 （nums[i]）的话，有 dp[0]中方法 凑成 容量为5的背包

        //那么凑整dp[5]有多少方法呢，也就是把 所有的 dp[j - nums[i]] 累加起来。
        //
        //所以求组合类问题的公式，都是类似这种：
        //
        //dp[j] += dp[j - nums[i]]
        for (int i = 0; i < nums.length; i++) {
            for (int j = x; j >= nums[i]; j--) {
                dp[j] += dp[j - nums[i]];
            }
        }
        return dp[x];
    }

    //518.零钱兑换II
    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                dp[j] += dp[j - coins[i]];
            }
        }
        return dp[amount];
    }

    //377. 组合总和 Ⅳ
    //该题为求出的排列数，即不同顺序也可以，需要先遍历背包再遍历物品
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 0; i <= target; i++) {
            for (int j = 0; j < nums.length; j++) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        return dp[target];
    }

    //322. 零钱兑换
    //该题为完全背包，即可以无限取nums[i],故内循环遍历顺序为正序
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 0; i < coins.length; i++) {
            for (int j = coins[i]; j <= amount; j++) {
                if (dp[j - coins[i]] != Integer.MAX_VALUE) {
                    dp[j] = Math.min(dp[j], dp[j - coins[i]] + 1);
                }
            }
        }
        return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
    }

    //279.完全平方数
    //组合问题，num[i]可以任取，故为内循环正序遍历
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        int max = Integer.MAX_VALUE;
        //从递归公式dp[j] = min(dp[j - i * i] + 1, dp[j]);
        // 可以看出每次dp[j]都要选最小的，所以非0下标的dp[j]一定要初始为最大值，
        // 这样dp[j]在递推的时候才不会被初始值覆盖。
        Arrays.fill(dp, max);
        //注意dp[0]的初始化要在填充数组之后
        dp[0] = 0;
        //遍历物品
        for (int i = 1; i * i <= n; i++) {
            //遍历背包
            for (int j = i * i; j <= n; j++) {
                if (dp[j - i * i] != max) {
                    dp[j] = Math.min(dp[j], dp[j - i * i] + 1);
                }
            }
        }
        return dp[n];
    }


    //139.单词拆分
    public boolean wordBreak(String s, List<String> wordDict) {
        //wordDict是物品，s是背包，
        //dp[i]表示取wordDict[0-j],能否凑成s的前i个
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        //因为可以任取，所以是正序遍历
        //因为是排列，所以背包在外循环，物品在内循环

        //注意背包的取值范围是0-s.length()!!!
        for (int i = 0; i <= s.length(); i++) {
            for (int j = 0; j < i; j++) {
                //s.substring是左闭右开
                String sub = s.substring(j, i);
                if (dp[j] && wordDict.contains(sub)) {
                    dp[i] = true;
                }
            }
        }
        return dp[s.length()];
    }


    //198.打家劫舍
    public int rob(int[] nums) {
        if (nums.length == 0 || nums == null) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        //dp[i]表示偷0-i以内的房屋，最大的价值
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);

        //{偷第i个房间、不偷第i个房间}选最大值
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }

    //121. 买卖股票的最佳时机
    public int maxProfit(int[] prices) {
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            //第i天持有股票的收益：第i-1天就持有，今天才买入
            dp[i][0] = Math.max(dp[i - 1][0], -prices[i]);
            //第i天不持有股票的收益：第i-1天就不持有，今天才卖出
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[prices.length - 1][1];
    }

    //122.买卖股票的最佳时机II 多次买卖
    public int maxProfit2(int[] prices) {
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            //第i天持有股票
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
        }
        return dp[prices.length - 1][1];
    }


    //123.买卖股票的最佳时机III 两次买卖
    public int maxProfit3(int[] prices) {
        int[][] dp = new int[prices.length][5];
        dp[0][0] = 0;
        dp[0][1] = -prices[0];
        dp[0][2] = 0;
        dp[0][3] = -prices[0];
        dp[0][4] = 0;
        for (int i = 1; i < prices.length; i++) {
            //0没有操作 （其实我们也可以不设置这个状态）
            //1第一次持有股票
            //2第一次不持有股票
            //3第二次持有股票
            //4第二次不持有股票
            dp[i][0] = dp[i - 1][0];
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
            dp[i][2] = Math.max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
            dp[i][3] = Math.max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
            dp[i][4] = Math.max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
        }
        return dp[prices.length - 1][4];
    }

    //188.买卖股票的最佳时机IV K次买卖
    public int maxProfit(int k, int[] prices) {
        //总共有2*k+1种状态
        int[][] dp = new int[prices.length][2 * k + 1];
        for (int i = 0; i < k; i++) {
            dp[0][2 * i + 1] = -prices[0];
        }
        for (int i = 1; i < prices.length; i++) {
            for (int j = 0; j < k; j++) {
                dp[i][2 * j + 1] = Math.max(dp[i - 1][2 * j + 1], dp[i - 1][2 * j] - prices[i]);
                dp[i][2 * j + 2] = Math.max(dp[i - 1][2 * j + 2], dp[i - 1][2 * j + 1] + prices[i]);
            }
        }
        return dp[prices.length - 1][2 * k];
    }

    //714.买卖股票的最佳时机含手续费
    public int maxProfit(int[] prices, int fee) {
        int[][] dp = new int[prices.length][2];
        dp[0][0] = -prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < prices.length; i++) {
            //持有股票
            dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
            //不持有股票
            dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee);
        }
        return Math.max(dp[prices.length - 1][0], dp[prices.length - 1][1]);
    }

    //300.最长递增子序列
    public int lengthOfLIS(int[] nums) {
        //dp[i]表示以nums[i]结尾的最长递增子序列
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int res = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    //dp[i]是dp[j]里面去最大值+1
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //674. 最长连续递增序列
    public int findLengthOfLCIS(int[] nums) {
        //dp[i]表示数组取以nums[i]结尾，最长连续递增子序列的长度为dp[i]
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        //最终的结果是dp[i]里最大的那个值
        int res = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] > nums[i - 1]) {
                dp[i] = dp[i - 1] + 1;
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //718. 最长重复子数组
    // 给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。
    public int findLength(int[] nums1, int[] nums2) {

        //dp[i][j]表示以数组A取nums1[i-1]，数组B取nums[j-1]结尾，A取到i-1，B取到j-1,
        //相当于在原来数组的基础上又加了一行和一列，dp[0][j]和dp[i][0]没有意义
        //为什么不去到nums1[i],nums[j]，因为这样需要初始化nums[0][j]和nums[i][0]，麻烦！！
        int[][] dp = new int[nums1.length + 1][nums2.length + 1];

        //最后的结果是所有的dp[i][j]里面取最大值
        int res = 0;
        for (int i = 1; i <= nums1.length; i++) {
            for (int j = 1; j <= nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;
    }

    //1143.最长公共子序列
    public int longestCommonSubsequence(String text1, String text2) {
        //dp[i][j]表示text1取[0,i-1]，text2取[0,j-1],以i-1,j-1结尾，最长公共子序列的长度，子序列可以不连续
        //相当于在原来数组的基础上又加了一行和一列，dp[0][j]和dp[i][0]没有意义
        //为什么不去到nums1[i],nums[j]，因为这样需要初始化nums[0][j]和nums[i][0]，麻烦！！
        int[][] dp = new int[text1.length() + 1][text2.length() + 1];
        //最后的结果是所有的dp[i][j]里面取最大值
        int res = 0;

        for (int i = 1; i <= text1.length(); i++) {
            for (int j = 1; j <= text2.length(); j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i][j - 1], dp[i - 1][j]);
                }
                res = Math.max(res, dp[i][j]);
            }
        }
        return res;
    }

    //53. 最大子序和
    public int maxSubArray(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }

        //dp[i]表示以下标i结尾，最大子序和为dp[i]
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            //dp[i - 1] + nums[i]，即：nums[i]加入当前连续子序列和
            //nums[i]，即：从头开始计算当前连续子序列和
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //392.判断子序列,判断s是否为t的子序列
    public boolean isSubsequence(String s, String t) {
        //dp[i][j]表示s以下标i-1结尾，t以下标j-1结尾，s和t的相同的子序列长度为dp[i][j],注意s是不能动的
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    //t删除下标为j-1的
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }
        return dp[s.length()][t.length()] == s.length();
    }

    //115.不同的子序列
    public int numDistinct(String s, String t) {
        //dp[i][j]表示s以下标i-1,t以下标j-1为结尾，s的子序列中t出现的个数为dp[i][j]
        int[][] dp = new int[s.length() + 1][t.length() + 1];
        for (int i = 0; i <= s.length(); i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i <= s.length(); i++) {
            for (int j = 1; j <= t.length(); j++) {
                if (s.charAt(i - 1) == t.charAt(j - 1)) {
                    //如果结尾元素相同,用该元素匹配+不用该元素匹配
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    //如果结尾元素不同,删除s的元素，继续匹配
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }
        return dp[s.length()][t.length()];
    }

    //583. 两个字符串的删除操作
    //给定两个单词 word1 和 word2 ，返回使得 word1 和  word2 相同所需的最小步数。
    public int minDistance(String word1, String word2) {
        //dp[i][j]表示word1以下标i-1为结尾，word2以下标j-1为结尾时，达到相等，所需删除的最少次数
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= word2.length(); j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1);
                }
            }
        }
        return dp[word1.length()][word2.length()];

    }

    //72. 编辑距离
    public int minDistance1(String word1, String word2) {
        //dp[i][j]表示word1以下标i-1为结尾，word2以下标j-1为结尾时，达到相等，所需操作的最少次数
        int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i <= word1.length(); i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j <= word2.length(); j++) {
            dp[0][j] = j;
        }

        for (int i = 1; i <= word1.length(); i++) {
            for (int j = 1; j <= word2.length(); j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1);
                    dp[i][j] = Math.min(dp[i][j], dp[i - 1][j - 1] + 1);
                }
            }
        }
        return dp[word1.length()][word2.length()];

    }

    //647. 回文子串
    public int countSubstrings(String s) {
        //dp[i][j]表示在区间范围[i,j]，s的子串是否为回文串
        int res = 0;
        boolean[][] dp = new boolean[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (j - i <= 1) {
                        dp[i][j] = true;
                        res++;
                    } else if (dp[i + 1][j - 1]) {
                        dp[i][j] = true;
                        res++;
                    }
                }
            }
        }
        return res;
    }

    //516.最长回文子序列
    public int longestPalindromeSubseq(String s) {
        //dp[i][j]表示字符串s在区间范围[i,j]，最长回文子序列的长度为dp[i][j]
        int[][] dp = new int[s.length()][s.length()];
        for (int i = s.length() - 1; i >= 0; i--) {
            for (int j = i; j < s.length(); j++) {
                if (s.charAt(i) == s.charAt(j)) {
                    if (i == j) {
                        dp[i][j] = 1;
                    } else if (j - i == 1) {
                        dp[i][j] = 2;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1] + 2;
                    }
                } else {
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[0][s.length() - 1];
    }

    //第77题. 组合
    LinkedList<Integer> path = new LinkedList<>();
    List<List<Integer>> res = new ArrayList<>();

    public List<List<Integer>> combine(int n, int k) {

        backtracking(n, k, 1);
        return res;

    }

    public void backtracking(int n, int k, int startIndex) {
        //递归终止的条件
        if (path.size() == k) {
            res.add(new ArrayList<>(path));//存放符合条件结果的集合
            return;
        }

        //列表中的元素个数 >= 还需要元素的个数
        //n-i+1>=k-path.size()
        for (int i = startIndex; i <= n + 1 - (k - path.size()); i++) {
            path.add(i); //处理节点
            backtracking(n, k, i + 1); //递归
            path.removeLast(); //回溯，撤销处理的节点
        }
    }

    //216.组合总和III
    public List<List<Integer>> combinationSum3(int k, int n) {
        LinkedList<Integer> path = new LinkedList<>();
        List<List<Integer>> res = new LinkedList<>();
        int startIndex = 1;
        int sum = 0;
        trackingCombinationSum3(k, n, startIndex, sum, path, res);
        return res;
    }

    public void trackingCombinationSum3(int k, int n, int startIndex, int sum, LinkedList<Integer> path, List<List<Integer>> res) {
        if (sum > n) {
            return;
        }
        if (path.size() == k) {//递归停止条件
            if (sum == n) {//递归停止条件
                res.add(new ArrayList<>(path));
                return;
            }
        }

        //startIndex表示每次循环起始位置
        for (int i = startIndex; i <= 9 - (k - path.size()) + 1; i++) {
            path.add(i);//处理节点
            sum += i;
            trackingCombinationSum3(k, n, i + 1, sum, path, res);//递归
            path.removeLast();//回溯，撤销处理的节点
            sum -= i;
        }
    }

    //17.电话号码的字母组合
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return res;
        }
        String[] numString = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        int num = 0;
        StringBuilder path = new StringBuilder();
        backLetterCombinations(digits, numString, num, path, res);
        return res;
    }

    public void backLetterCombinations(String digits, String[] numString, int num, StringBuilder path, List<String> res) {
        if (path.length() == digits.length()) {
            res.add(path.toString());
            return;
        }
        String str = numString[digits.charAt(num) - '0'];
        for (int i = 0; i < str.length(); i++) {
            path.append(str.charAt(i));
            backLetterCombinations(digits, numString, num + 1, path, res);
            path.deleteCharAt(path.length() - 1);
        }
    }

    LinkedList<Integer> pathCom = new LinkedList<>();
    List<List<Integer>> resCom = new ArrayList<>();

    //39. 组合总和，可以重复选
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int sum = 0;
        int startIndex = 0;
        trackCom(candidates, target, startIndex, sum);
        return resCom;
    }

    public void trackCom(int[] candidates, int target, int startIndex, int sum) {

        if (sum > target) {
            return;
        }
        if (sum == target) {
            resCom.add(new ArrayList<>(pathCom));
        }

        for (int i = startIndex; i < candidates.length; i++) {
            pathCom.add(candidates[i]);//处理节点
            sum += candidates[i];
            //一个集合，组合问题，可以重复选，故startIndex传入i
            trackCom(candidates, target, i, sum);//递归
            pathCom.removeLast();
            sum -= candidates[i];
        }
    }

    //40.组合总和II
    LinkedList<Integer> pathCom2 = new LinkedList<>();
    List<List<Integer>> resCom2 = new ArrayList<>();

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        Arrays.sort(candidates);
        int startIndex = 0;
        int sum = 0;
        combinationSum2Helper(candidates, target, startIndex, sum);
        return resCom2;
    }

    public void combinationSum2Helper(int[] candidates, int target, int startIndex, int sum) {

        if (sum == target) {
            resCom2.add(new ArrayList<>(pathCom2));
            return;
        }
        for (int i = startIndex; i < candidates.length; i++) {

            pathCom2.add(candidates[i]);
            sum += candidates[i];
            combinationSum2Helper(candidates, target, startIndex + 1, sum);
            pathCom2.removeLast();
            sum -= candidates[i];
        }
    }

    LinkedList<Integer> pathSub = new LinkedList<>();
    List<List<Integer>> resSub = new ArrayList<>();

    //78.子集
    public List<List<Integer>> subsets(int[] nums) {
        int startIndex = 0;
        trackSubsets(nums, startIndex);
        return resSub;
    }

    public void trackSubsets(int[] nums, int startIndex) {
        resSub.add(new ArrayList<>(pathSub));
        if (startIndex >= nums.length) {
            return;
        }

        for (int i = startIndex; i < nums.length; i++) {
            pathSub.add(nums[i]);
            trackSubsets(nums, i + 1);
            pathSub.removeLast();
        }
    }

    LinkedList<Integer> pathSubN = new LinkedList<>();
    List<List<Integer>> resSubN = new ArrayList<>();
    boolean[] usedSub;

    //90.子集II
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        int startIndex = 0;
        Arrays.sort(nums);
        usedSub = new boolean[nums.length];
        trackSubsetsWithDup(nums, startIndex);
        return resSubN;
    }

    public void trackSubsetsWithDup(int[] nums, int startIndex) {
        resSubN.add(new ArrayList<>(pathSubN));
        if (startIndex >= nums.length) {
            return;
        }

        for (int i = startIndex; i < nums.length; i++) {
            if (i > 0 && nums[i] == nums[i - 1] && usedSub[i - 1] == false) {
                continue;
            }
            pathSubN.add(nums[i]);
            usedSub[i] = true;
            trackSubsetsWithDup(nums, i + 1);
            usedSub[i] = false;
            pathSubN.removeLast();
        }
    }

    LinkedList<Integer> pathSeq = new LinkedList<>();
    List<List<Integer>> resSeq = new ArrayList<>();

    //491.递增子序列
    public List<List<Integer>> findSubsequences(int[] nums) {
        int startIndex = 0;
        findSubsequencesHelper(nums, startIndex);
        return resSeq;
    }

    public void findSubsequencesHelper(int[] nums, int startIndex) {
        if (pathSeq.size() > 1) {
            resSeq.add(new ArrayList<>(pathSeq));
        }
        HashSet<Integer> set = new HashSet<>();
        for (int i = startIndex; i < nums.length; i++) {
            if ((!pathSeq.isEmpty() && nums[i] < pathSeq.get(pathSeq.size() - 1)) || set.contains(nums[i])) {
                continue;
            }
            pathSeq.add(nums[i]);
            set.add(nums[i]);
            findSubsequencesHelper(nums, i + 1);
            pathSeq.removeLast();
            set.remove(set.size() - 1);
        }
    }

    //46.全排列
    LinkedList<Integer> pathPer = new LinkedList<>();
    List<List<Integer>> resPer = new ArrayList<>();

    public List<List<Integer>> permute(int[] nums) {
        permuteHelper(nums);
        return resPer;
    }

    public void permuteHelper(int[] nums) {
        if (pathPer.size() == nums.length) {
            resPer.add(new ArrayList<>(pathPer));
            return;
        }

        for (int i = 0; i < nums.length; i++) {
            if (pathPer.contains(nums[i])) {
                continue;
            }
            pathPer.add(nums[i]);
            permuteHelper(nums);
            pathPer.removeLast();
        }
    }

    //47.全排列 II
    LinkedList<Integer> pathUni = new LinkedList<>();
    List<List<Integer>> resUni = new ArrayList<>();
    boolean[] usedUni;

    public List<List<Integer>> permuteUnique(int[] nums) {
        Arrays.sort(nums);
        usedUni = new boolean[nums.length];
        permuteUniqueHelper(nums);
        return resUni;
    }

    public void permuteUniqueHelper(int[] nums) {
        if (pathUni.size() == nums.length) {
            resUni.add(new ArrayList<>(pathUni));
        }

        for (int i = 0; i < nums.length; i++) {
            //去除重复元素
            if (i > 0 && nums[i] == nums[i - 1] && usedUni[i - 1] == false) {
                continue;
            }
            if (usedUni[i] == true) {
                continue;
            }
            usedUni[i] = true;
            pathUni.add(nums[i]);
            permuteUniqueHelper(nums);
            usedUni[i] = false;
            pathUni.removeLast();

        }
    }

    //455. 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        //大饼干既可以满足大胃口的，也可以满足小胃口的，满足小胃口的就浪费了，优先满足大胃口的
        Arrays.sort(g);
        Arrays.sort(s);
        int res = 0;
        int index = s.length - 1;//饼干最大值下标

        //遍历胃口
        for (int i = g.length - 1; i >= 0; i--) {
            if (index >= 0 && s[index] >= g[i]) {
                res++;
                index--;
            }
        }

        return res;
    }

    //53. 最大子序和
    public int maxSubArray1(int[] nums) {

        int res = nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //贪心解法
    public int maxSubArray2(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int res = nums[0];
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (sum > res) {
                res = sum;
            }
            if (sum <= 0) {
                sum = 0;
            }
        }
        return res;
    }

    //55. 跳跃游戏
    public boolean canJump(int[] nums) {
        int cover = 0;
        for (int i = 0; i <= cover; i++) {
            cover = Math.max(cover, i + nums[i]);
            if (cover >= nums.length - 1) {
                return true;
            }
        }
        return false;
    }

    //1005.K次取反后最大化的数组和
    public int largestSumAfterKNegations(int[] nums, int k) {
        Integer[] temp = new Integer[nums.length];
        for (int i = 0; i < nums.length; i++) {
            temp[i] = nums[i];
        }
        Arrays.sort(temp, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Math.abs(o2) - Math.abs(o1);
            }
        });

        for (int i = 0; i < temp.length; i++) {
            if (temp[i] < 0 && k > 0) {
                temp[i] = -temp[i];
                k--;
            }
        }
        if (k % 2 == 1) {
            temp[temp.length - 1] = -temp[temp.length - 1];
        }
        int sum = 0;
        for (int i = 0; i < temp.length; i++) {
            sum += temp[i];
        }
        return sum;
    }

    //135. 分发糖果
    public int candy(int[] ratings) {
        int[] res = new int[ratings.length];
        Arrays.fill(res, 1);
        for (int i = 1; i < ratings.length; i++) {
            if (ratings[i] > ratings[i - 1]) {
                res[i] = res[i - 1] + 1;
            }
        }

        for (int i = ratings.length - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                res[i] = Math.max(res[i + 1] + 1, res[i]);
            }
        }
        int sum = 0;
        for (int i = 0; i < res.length; i++) {
            sum += res[i];
        }
        return sum;
    }

    //406.根据身高重建队列
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, (p1, p2) -> {
            if (p2[0] != p1[0]) {
                //身高从大到小排序
                return p2[0] - p1[0];
            } else {
                //位次从小到大排序
                return p1[1] - p2[1];
            }
        });
        LinkedList<int[]> res = new LinkedList<>();
        for (int i = 0; i < people.length; i++) {
            res.add(people[i][1], people[i]);
        }
        return res.toArray(new int[people.length][]);
    }

    // 452. 用最少数量的箭引爆气球
    public int findMinArrowShots(int[][] points) {
        //更新共用的最小右边界
        //按照左边界进行排序
        Arrays.sort(points, (a, b) -> {
            if (a[0] > b[0]) {
                return 1;
            } else {
                return -1;
            }
        });
        int res = 1;
        for (int i = 1; i < points.length; i++) {
            //如果该元素的左边界>上一个元素的右边界，则res+1
            if (points[i][0] > points[i - 1][1]) {
                res++;
            } else {
                //如果如果该元素的左边界<上一个元素的右边界（右边界需要不断更新，且是同一个组里面共用的右边界）
                points[i][1] = Math.min(points[i][1], points[i - 1][1]);
            }
        }
        return res;
    }

    //435. 无重叠区间
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> {
            if (a[0] > b[0]) {
                return 1;
            } else if (a[0] == b[0]) {
                return 0;
            } else {
                return -1;
            }
        });
        int res = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] < intervals[i - 1][1]) {
                res++;
                intervals[i][1] = Math.min(intervals[i][1], intervals[i - 1][1]);
            } else {

            }
        }
        return res;
    }

    //56. 合并区间
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> {
            return Integer.compare(a[0], b[0]);
        });
        int left = intervals[0][0];
        int right = intervals[0][1];
        List<int[]> res = new ArrayList<>();
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[i][0] > right) {
                res.add(new int[]{left, right});
                left = intervals[i][0];
                right = intervals[i][1];
            } else {
                right = Math.max(right, intervals[i][1]);
            }
        }
        res.add(new int[]{left, right});
        return res.toArray(new int[res.size()][]);
    }

    //199.二叉树的右视图
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.peek();
                queue.poll();
                if (i == size - 1) {
                    res.add(tmpNode.val);
                }

                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
        }
        return res;
    }

    //637.二叉树的层平均值
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0.0;
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                sum += tmpNode.val;

                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
            double avg = sum / size;
            res.add(avg);
        }
        return res;

    }

    //429.N叉树的层序遍历
//    public List<List<Integer>> levelOrder(Node root) {
//        List<List<Integer>> res = new ArrayList<>();
//        if (root == null) {
//            return res;
//        }
//        Queue<Node> queue = new LinkedList<>();
//        queue.add(root);
//        while (!queue.isEmpty()) {
//            int size = queue.size();
//            List<Integer> tmpList = new ArrayList<>();
//            for (int i = 0; i < size; i++) {
//                Node tmpNode = queue.poll();
//                tmpList.add(tmpNode.val);
//                if (tmpNode.children != null) {
//                    int len = tmpNode.children.size();
//                    for (int j = 0; j < len; j++) {
//                        queue.add(tmpNode.children.get(j));
//                    }
//                }
//            }
//            res.add(tmpList);
//        }
//        return res;
//    }

    //515.在每个树行中找最大值
    public List<Integer> largestValues(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int max = Integer.MIN_VALUE;
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                max = Math.max(max, tmpNode.val);
                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
            res.add(max);
        }
        return res;

    }

    //116.填充每个节点的下一个右侧节点指针
    public Node connect(Node root) {
        if (root == null) {
            return null;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);

        while (!queue.isEmpty()) {
            int size = queue.size();
            Node pre = queue.poll();
            if (pre.left != null) {
                queue.add(pre.left);
            }
            if (pre.right != null) {
                queue.add(pre.right);
            }
            for (int i = 1; i < size; i++) {
                Node cur = queue.poll();
                pre.next = cur;
                pre = cur;

                if (cur.left != null) {
                    queue.add(cur.left);
                }
                if (cur.right != null) {
                    queue.add(cur.right);
                }
            }
        }
        return root;
    }

    //104.二叉树的最大深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
            res++;
        }
        return res;
    }

    //111.二叉树的最小深度
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            res++;
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                if (tmpNode.left == null && tmpNode.right == null) {
                    return res;
                }

                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }

        }
        return res;
    }

    //226.翻转二叉树
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        invertTree(root.left);
        invertTree(root.right);
        swapNode(root);
        return root;
    }

    public TreeNode invertTree1(TreeNode root) {
        if (root == null) {
            return null;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                swapNode(tmpNode);
                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
        }
        return root;
    }


    private void swapNode(TreeNode root) {
        TreeNode tmpNode = root.left;
        root.left = root.right;
        root.right = tmpNode;
    }

    //101. 对称二叉树
    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return isMirror(root.left, root.right);
    }

    public boolean isMirror(TreeNode node1, TreeNode node2) {

        if (node1 == null && node2 != null) {
            return false;
        }
        if (node1 != null && node2 == null) {
            return false;
        }
        if (node1 == null && node2 == null) {
            return true;
        }

        //如果node1这棵树的左子节点、node2这棵树的右子节点是对称的，
        //且node1这棵树的右子节点、node2这棵树的左子结点是对称的，
        //且node1.val==node2.val，那么node1和node2就是对称的
        return isMirror(node1.left, node2.right) && isMirror(node1.right, node2.left) && node1.val == node2.val;
    }

    //110.平衡二叉树
    public boolean isBalanced(TreeNode root) {
        return getHeight(root) == -1 ? false : true;
    }

    public int getHeight(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftHeight = getHeight(node.left);
        if (leftHeight == -1) {
            return -1;
        }
        int rightHeight = getHeight(node.right);
        if (rightHeight == -1) {
            return -1;
        }

        if (Math.abs(leftHeight - rightHeight) > 1) {
            return -1;
        } else {
            return 1 + Math.max(leftHeight, rightHeight);
        }
    }

    List<String> treeRes = new LinkedList<>();
    LinkedList<Integer> treePath = new LinkedList<>();

    //257. 二叉树的所有路径
    public List<String> binaryTreePaths(TreeNode root) {
        treePathHelper(root);
        return treeRes;
    }

    public void treePathHelper(TreeNode node) {
        treePath.add(node.val);
        if (node.left == null && node.right == null) {
            StringBuffer tmp = new StringBuffer();
            for (int i = 0; i < treePath.size() - 1; i++) {
                tmp.append(treePath.get(i));
                tmp.append("->");
            }
            tmp.append(treePath.get(treePath.size() - 1));
            treeRes.add(tmp.toString());
        }

        if (node.left != null) {
            treePathHelper(node.left);
            treePath.removeLast();
        }
        if (node.right != null) {
            treePathHelper(node.right);
            treePath.removeLast();
        }
    }

    //404.左叶子之和
    public int sumOfLeftLeaves1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int sum = 0;
        if (root.left != null && root.left.left == null && root.left.right == null) {
            sum += root.left.val;
        }

        return sumOfLeftLeaves1(root.left) + sumOfLeftLeaves1(root.right) + sum;


    }


    public int sumOfLeftLeaves(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();

                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                    if (tmpNode.left.left == null && tmpNode.left.right == null) {
                        res += tmpNode.left.val;
                    }
                }

                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
        }
        return res;
    }

    //513.找树左下角的值
    public int findBottomLeftValue(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode tmpNode = queue.poll();
                if (i == 0) {
                    res = tmpNode.val;
                }
                if (tmpNode.left != null) {
                    queue.add(tmpNode.left);
                }
                if (tmpNode.right != null) {
                    queue.add(tmpNode.right);
                }
            }
        }
        return res;
    }

    //112. 路径总和
//    LinkedList<Integer> path1 = new LinkedList();
    int sum1 = 0;
    int res1 = 0;

    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) {
            return false;
        }
        hasPathSumHelper(root, targetSum);
        return res1 != 0;
    }

    public void hasPathSumHelper(TreeNode root, int targetSum) {
        if (root.left == null && root.right == null) {
            sum1 += root.val;
            if (sum1 == targetSum) {
                res1++;
            }
            return;
        }

        sum1 += root.val;
        if (root.left != null) {
            hasPathSumHelper(root.left, targetSum);
            sum1 -= root.left.val;
        }
        if (root.right != null) {
            hasPathSumHelper(root.right, targetSum);
            sum1 -= root.right.val;
        }

    }

    //98.验证二叉搜索树
    //遇到二叉搜索树要想到两点：
    //1.根节点和左右节点的大小关系，以及左右子树也是二叉搜索树
    //2.中序遍历的结果，就是从小到大排的
    TreeNode preNode = null;

    public boolean isValidBST(TreeNode root) {
        if (root == null) {
            return true;
        }
        boolean left = isValidBST(root.left);
        if (left == false) {
            return false;
        }
        if (preNode != null && preNode.val > root.val) {
            return false;
        }
        preNode = root;
        boolean right = isValidBST(root.right);
        if (right == false) {
            return false;
        }
        return true;

    }

    //977.有序数组的平方
    public int[] sortedSquares(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        int[] res = new int[nums.length];
        int index = nums.length - 1;
        while (j >= i) {
            int A = nums[i] * nums[i];
            int B = nums[j] * nums[j];
            if (A >= B) {
                res[index--] = A;
                i++;
            } else {
                res[index--] = B;
                j--;
            }
        }
        return res;
    }

    //59.螺旋矩阵II
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int left = 0;
        int right = n - 1;
        int up = 0;
        int down = n - 1;
        int index = 1;
        while (index <= n * n) {
            if (up <= down) {
                for (int i = left; i <= right; i++) {
                    res[up][i] = index++;
                }
                up++;
            }

            if (left <= right) {
                for (int i = up; i <= down; i++) {
                    res[i][right] = index++;
                }
                right--;
            }

            if (up >= down) {
                for (int i = right; i >= left; i--) {
                    res[down][i] = index++;
                }
                down--;
            }

            if (left <= right) {
                for (int i = down; i >= up; i--) {
                    res[i][left] = index++;
                }
                left++;
            }
        }
        return res;
    }

    //203.移除链表元素
    public ListNode removeElements(ListNode head, int val) {
        if (head == null) {
            return null;
        }
        ListNode dummy = new ListNode(-1, head);
        ListNode cur = head;
        ListNode pre = dummy;
        while (cur != null) {
            if (cur.val == val) {
                pre.next = cur.next;
            } else {
                pre = cur;
            }
            cur = cur.next;
        }
        return dummy.next;

    }

    //206.反转链表
    public ListNode reverseList(ListNode head) {
        ListNode pre = head;
        ListNode cur = head;
        while (cur != null) {

        }
        return null;
    }


    public static void main(String[] args) {
        int[] nums = new int[]{-4, -1, 0, 3, 10};
        Solution solution = new Solution();
        solution.sortedSquares(nums);


        TreeNode node5 = new TreeNode(5);
        TreeNode node4 = new TreeNode(4);
        TreeNode node8 = new TreeNode(8);
        TreeNode node11 = new TreeNode(11);
        TreeNode node13 = new TreeNode(13);
        TreeNode node3 = new TreeNode(3);
        TreeNode node7 = new TreeNode(7);
        TreeNode node2 = new TreeNode(2);
        TreeNode node1 = new TreeNode(1);
        node5.left = node4;
        node5.right = node8;
        node4.left = node11;
        node11.left = node7;
        node11.right = node2;
        node8.left = node13;
        node8.right = node3;
        node4.right = node1;

        int sum = solution.sumOfLeftLeaves1(node5);
        System.out.println(sum);

//        boolean b = solution.hasPathSum(node5,22);
//        System.out.println(b);

    }


    //test占位
    public int test(int[] nums1, int[] nums2) {
        for (int i = 0; i < 100; i++) {

        }

        for (int i = 0; i < 100; i++) {

        }

        for (int i = 0; i < 100; i++) {

        }
        return 0;
    }


}
