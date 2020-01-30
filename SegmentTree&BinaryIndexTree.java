lc 307: Range Sum Array Query -- Mutable

Segment Tree ： 
// 线段树 segment tree详细讲解： http://blog.csdn.net/sunny606/article/details/38817393
// 树状数组详细讲解：            http://blog.csdn.net/qq_18661257/article/details/47347995
public class NumArray {
    class SegmentTreeNode{
        int start;
        int end;
        SegmentTreeNode left;
        SegmentTreeNode right;
        int sum;
        public SegmentTreeNode(int start, int end){
            this.start = start;
            this.end = end;
            this.left = null;
            this.right = null;
            this.sum = 0;            
        }
    }
    SegmentTreeNode root;
    int[] nums;
    public NumArray(int[] nums) {
        this.root = buildTree(nums, 0, nums.length - 1);
        this.nums = nums;
    }
    
    public SegmentTreeNode buildTree(int[] nums, int start, int end){       
        if(start > end){                                                    // 这个也要写,只是为了解决nums.length == 0 这一个testcase
            return null;
        }
        SegmentTreeNode cur = new SegmentTreeNode(start, end);
        if(start == end){                                                   // 这个要写,因为永远不会跳到start > end, 往左走更新end时,end最小也就是等于start
            cur.sum = nums[start];
        }else{
            int mid = start + (end - start) / 2;
            cur.left = buildTree(nums, start, mid);
            cur.right = buildTree(nums, mid + 1, end);
            cur.sum = cur.left.sum + cur.right.sum; 
        }              
        return cur;
    }
    
    public void update(int i, int val) {
        update(root, i, val);
    }
    
    private void update(SegmentTreeNode root, int i, int val){              // 如果不把这个设为helper函数，那么下次使用root的时候root是null;
    	int diff = val - nums[i];
        this.nums[i] = val;
        while(root != null){
            root.sum += diff;
            int mid = root.start + (root.end - root.start) / 2;
            if(i <= mid){
                root = root.left;
            }else{
                root = root.right;
            }
        }
    }
    
    public int sumRange(int i, int j) {
        return sumRangeHelp(root, i, j);
    }
    
    private int sumRangeHelp(SegmentTreeNode root, int i, int j){   
        if(root.start == i && root.end == j){
            return root.sum;
        }   
        int mid = root.start + (root.end - root.start) / 2;
        if(mid < i){
            return sumRangeHelp(root.right, i, j);
        }else if(mid >= j){
            return sumRangeHelp(root.left, i, j);
        }else{
            return sumRangeHelp(root.left, i, mid) + sumRangeHelp(root.right, mid + 1, j);
        }
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(i,val);
 * int param_2 = obj.sumRange(i,j);
 */


Binary Index Tree:

这个树状数组比较有意思，所有的奇数位置的数字和原数组对应位置的相同，偶数位置是原数组若干位置之和，假如原数组A(a1, a2, a3, a4 ...)，
和其对应的树状数组C(c1, c2, c3, c4 ...)有如下关系：
C1 = A1
C2 = A1 + A2
C3 = A3
C4 = A1 + A2 + A3 + A4
C5 = A5
C6 = A5 + A6
C7 = A7
C8 = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8
...
那么是如何确定某个位置到底是有几个数组成的呢，原来是根据坐标的最低位Low Bit来决定的，所谓的最低位，就是二进制数的最右边的一个1开始，
加上后面的0(如果有的话)组成的数字，例如1到8的最低位如下面所示：
坐标          二进制          最低位

1               0001          1

2               0010          2

3               0011          1

4               0100          4

5               0101          1

6               0110          2

7               0111          1

8               1000          8

...

最低位的计算方法有两种，一种是 x&(x^(x–1))，另一种是利用补码特性 x&-x。

这道题我们先根据给定输入数组建立一个树状数组bit，然后更新某一位数字时，根据最低位的值来更新后面含有这一位数字的地方，一般只需要更新部分偶数位置的值即可，
在计算某一位置的前缀和时，利用树状数组的性质也能高效的算出来：

//implementation 1
public class NumArray {
    int[] nums;
    int[] BIT;
    
    public NumArray(int[] nums) {
        this.nums = new int[nums.length];         // 这里没有写this.nums = nums,而是重新开了一个数组,因为如果直接赋值,那么我们后面update的时候见不了BIT array
        BIT = new int[nums.length + 1];
        for(int i = 0; i < nums.length; i ++){
            update(i, nums[i]);                   // constructor里的update是为了建立BIT array
        }
    }
    
    public void update(int i, int val) {          // 逐层更新BIT array, 最后顺便把nums[i]也改了
        int diff = val - nums[i];
        for(int index = i + 1; index < BIT.length; index += (index & -index)){    // index += (index & -index)实际上相当于挪到parent node
            BIT[index] += diff;
        }
        nums[i] = val;
    }
    
    public int sumRange(int i, int j) {
         return getSum(j + 1) - getSum(i);
    }
    
    private int getSum(int index){               // 求 nums[1]--nums[index] 的和,前缀和
        int result = 0;
        for(int i = index; i > 0; i -= (i & -i)){  // i -= (i & -i)实际相当于往前倒，跨过已经sum过的range.
            result += BIT[i];
        }
        return result;
    }
}

************************************
****一维树状数组 单点更新 区间求值****  1D Range Query, Point Update
************************************
lc 307: Range Sum Query - Mutable

//implementation 2:  保证array和BIT长度一样，这样map index的时候好理解一些
class NumArray {
    
    int[] BIT;
    int[] array;
    
    public NumArray(int[] nums) {
        this.array = new int[nums.length + 1];
        this.BIT = new int[nums.length + 1];
        for(int i = 0; i < nums.length; i ++){
            update(i, nums[i]);
        }
    }
    
    public void update(int i, int val) {
        int diff = val - array[i + 1];
        for(int index = i + 1; index < BIT.length; index += (index & -index)){
            BIT[index] += diff;
        }
        array[i + 1] = val;
    }
    
    public int sumRange(int i, int j) {
        return getSum(j + 1) - getSum(i);
    }
    
    private int getSum(int i){
        int result = 0;
        for(int index = i; index > 0; index -= (index & -index)){
            result += BIT[index];
        }
        return result;
    }
}

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray obj = new NumArray(nums);
 * obj.update(i,val);
 * int param_2 = obj.sumRange(i,j);
 */
************************************ 
****一维树状数组 区间更新 单点求值****  1D Range Update, Point Query
************************************
参考资料: https://www.geeksforgeeks.org/binary-indexed-tree-range-updates-point-queries/
先看method2,才能理解这个方法***重点method2!!!!

public class Solution {
    public static void main(String[] args) {
        Solution bit = new Solution(5);
        bit.update(1, 4, 1);
        bit.update(2, 3, 2);
        System.out.println(bit.get(5));
    }
    int[] bit;
    public Solution(int n){
        bit = new int[n + 1];
    }
    public void update(int left, int right, int val){
        update(left, val);
        update(right + 1, -val);
    }
    
    public void update(int index, int val){
        while(index < bit.length){
            bit[index] += val;
            index += index & (-index);
        }
    }
    
    public int get(int index){
        int result = 0;
        while(index > 0){
            result += bit[index];
            index -= index & (-index);
        }
        return result;
    }
}

************************************
****二维树状数组 单点更新 区间求值****    2D Range Query, Point Update
************************************
lc 308: Range Sum Query 2D - Mutable

Binary Index Tree:

public class NumMatrix {
    int[][] tree;
    int[][] nums;
    public NumMatrix(int[][] matrix) {
        int m = matrix.length;
        if(m == 0){
            return;
        }
        int n = matrix[0].length;
        tree = new int[m + 1][n + 1];
        nums = new int[m][n];
        for(int i = 0; i < m; i ++){
            for(int j = 0; j < n; j ++){
                update(i, j, matrix[i][j]);
            }
        }
    }
    
    public void update(int row, int col, int val) {
        int diff = val - nums[row][col];
        for(int i = row + 1; i < tree.length; i += (i & -i)){
            for(int j = col + 1; j < tree[0].length; j += (j & -j)){
                tree[i][j] += diff;
            }
        }
        nums[row][col] = val;
    }
    
    public int sumRegion(int row1, int col1, int row2, int col2) {
        return getSum(row2, col2) - getSum(row1 - 1, col2) - getSum(row2, col1 - 1) + getSum(row1 - 1, col1 - 1);
    }
    
    private int getSum(int row, int col){
        if(row < 0 || col < 0){
            return 0;
        }
        int result = 0;
        for(int i = row + 1; i > 0; i -= (i& -i)){
            for(int j = col + 1; j > 0; j -= (j & -j)){
                result += tree[i][j];
            }
        }
        return result;
    }
}

/**
 * Your NumMatrix object will be instantiated and called as such:
 * NumMatrix obj = new NumMatrix(matrix);
 * obj.update(row,col,val);
 * int param_2 = obj.sumRegion(row1,col1,row2,col2);
 */

************************************ 
****二维树状数组 区间更新 单点求值****  2D Range Update, Point Query
************************************
思路跟一维Range update, Point Query一样,只是加了一个维度
更新操作的时候,首先要了解更新操作的实质: 当更新某一个point(i,j)的值时,实际上更新了(i,j)~(m,n)的所有值,这个方块里的element全部都+val
所以我们要实现只更新某一个区域里的值时,假设区域左上角是(x1,y1),右下角是(x2,y2)
我们要更新(x1, y1) + value, (x1, y2 + 1) - value, (x2 + 1, y1) - value, (x2 + 1, y2 + 1) + value.
类似于immutable 2D query时求一个矩阵的面积的思路. 只不过那个是往左上看,这个是往右下看

public class Solution {
    
    public static void main(String[] args) {
        Solution bit = new Solution(6,6);
        // 第一次区域： 左上角(1,1) 右下角(2,2)
        // 另外这个题中定义的矩阵是,左上角的坐标是(0,0),向下x增加,向右y增加

        // bit.update(1, 1, 1);        
        // bit.update(1, 3, -1);
        // bit.update(3, 1, -1);
        // bit.update(3, 3, 1);
        // Wrap 4 updates above into one update function
        bit.update(1,1,2,2,1);
        
        // 第2次区域： 左上角(1,2) 右下角(3,4)
        // bit.update(1, 2, 2);        
        // bit.update(1, 5, -2);
        // bit.update(4, 2, -2);
        // bit.update(4, 5, 2);
        bit.update(1,2,3,4,2);
        System.out.println(bit.get(2,2));
    }   
    
    int[][] bit;
    public Solution(int m, int n){
        bit = new int[m + 1][n + 1];
    }
    
    public void update(int x1, int y1, int x2, int y2, int val){
        update(x1, y1, val);
        update(x1, y2 + 1, -val);
        update(x2 + 1, y1, -val);
        update(x2 + 1, y2 + 1, val);
    }

    public void update(int row, int col, int val){
        for(int i = row + 1; i < bit.length; i += i & (-i)){
            for(int j = col + 1; j < bit.length; j += j & (-j)){
                bit[i][j] += val;
            }
        }
    }
    
    public int get(int row, int col){
        int result = 0;
        for(int i = row + 1; i > 0; i -= i & (-i)){
            for(int j = col + 1; j > 0; j -= j & (-j)){
                result += bit[i][j];
            }
        }
        return result;
    }
}

************************************ 
****一维树状数组 区间更新 单点求值****  1D Range Update, Range Query
************************************

参考资料：https://stackoverflow.com/questions/27875691/need-a-clear-explanation-of-range-updates-and-range-queries-binary-indexed-tree/27877427#27877427
举个例子：https://cs.stackexchange.com/questions/33014/range-update-range-query-with-binary-indexed-trees

public class Solution {
    public static void main(String[] args) {
        Solution bit = new Solution(10);
        bit.rangeUpdate(2, 5, 2);
        System.out.println(bit.rangeSum(3, 5));
    }
    int[] bit1;
    int[] bit2;
    public Solution(int n){
        bit1 = new int[n + 1];
        bit2 = new int[n + 1];
    }
    
    public void rangeUpdate(int i, int j, int val){
        update(bit1, i, val);
        update(bit1, j + 1, -val);
        
        update(bit2, i, (i - 1) * val);
        update(bit2, j + 1, -(i - 1) * val - (j - i + 1) * val);
    }
    
    public void update(int[] bit, int i, int val){
        i ++;
        while(i < bit.length){
            bit[i] += val;
            i += i & (-i);
        }
    }
    
    public int rangeSum(int i, int j){
        return getSum(bit1, i) * i - getSum(bit2, i);
    }
    
    public int getSum(int[] bit, int i){
        int result = 0;
        i ++;
        while(i > 0){
            result += bit[i];
            i -= i & (-i);
        }
        return result;
    }
}