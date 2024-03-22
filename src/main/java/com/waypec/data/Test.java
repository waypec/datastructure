package com.waypec.data;

import java.util.List;

public class Test {
    public volatile int inc = 0;

    public void increase() {
        inc++;
    }

    public static void main(String[] args) {
        final Test test = new Test();
        for (int i = 0; i < 10; i++) {
            new Thread() {
                public void run() {
                    for (int j = 0; j < 1000; j++)
                        test.increase();
                }

                ;
            }.start();
        }
        while (Thread.activeCount() > 2)
            Thread.yield();
        System.out.println(test.inc);



        Solution solution = new Solution();

        String a = "hello";
        System.out.println(a.substring(0,3));

        int[] nums = new int[]{4,0,3,2,5};
        List<List<Integer>> res = solution.permute(nums);
        System.out.println(res);

        int res1 = solution.maxSubArray1(new int[]{-2, 1, -3, 4, -1, 2, 1, -5, 4});
        System.out.println(res1);

    }
}