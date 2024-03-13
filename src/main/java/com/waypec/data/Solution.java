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

        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[nums.length - 1];
    }
}
