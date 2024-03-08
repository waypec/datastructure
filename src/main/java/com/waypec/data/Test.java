package com.waypec.data;

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
        int[] nums = new int[]{4,2,0,3,2,5};

        solution.trap(nums);
    }
}