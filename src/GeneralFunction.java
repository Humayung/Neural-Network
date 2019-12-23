import java.util.ArrayList;
import java.util.Random;


public class GeneralFunction {
    static Random random = new Random();

    public static int[] subset(int[] arr, int start, int end) {
        end = end == 0 ? arr.length : end;
        end = end < 0 ? arr.length + end : end;

        int newLength = end - start;
        int[] newArr = new int[newLength];
        System.arraycopy(arr, start, newArr, 0, newLength);
        return newArr;
    }

    public static float max(float[] arr){
        float max = arr[0];
        for(float f : arr){
            max = Math.max(f, max);
        }
        return max;
    }

    public static float min(float[] arr){
        float min = arr[0];
        for(float f : arr){
            min = Math.min(f, min);
        }
        return min;
    }

    public static int argmax(float[] arr){
        // FUCK THIS!!!!
        float max = arr[0];
        int imax = 0;
        for(int i = 0; i < arr.length; i++){
            if(max < arr[i]) {
                max = arr[i];
                imax = i;
            }
        }
        return imax;
    }
    public static int argmin(float[] arr){
        float min = arr[0];
        int imin = 0;
        for(int i = 0; i < arr.length; i++){
            if(min > arr[i]) {
                min = arr[i];
                imin = i;
            }
        }
        return imin;
    }

    public static String[] subset(String[] arr, int start, int end) {
        end = end == 0 ? arr.length : end;
        end = end < 0 ? arr.length + end : end;

        int newLength = end - start;
        String[] newArr = new String[newLength];
        System.arraycopy(arr, start, newArr, 0, newLength);
        return newArr;
    }

    public static int[][] zip(int[] a, int[] b) {
        int length = Math.min(a.length,
                b.length);
        int[][] arr = new int[length][2];
        for (int i = 0; i < length; i++) {
            arr[i][0] = a[i];
            arr[i][1] = b[i];
        }
        return arr;
    }

    public static Matrix sq(Matrix m) {
        return m.multiply(m);
    }

    public static float sq(float val) {
        return val * val;
    }

    public static Tensor mean(Matrix m) {
        float total = 0;
        for (Tensor[] arr : m.contents){
            total += mean(arr).getData();
        }
        return Tensor.param(total/m.contents.length);
    }

    public static Tensor mean(Tensor[] arr) {
        float total = 0;
        for (Tensor f : arr) {
            total += f.getData();
        }
        return Tensor.param(total / arr.length);
    }

    public static float[] mult(float[] arr, float c){
        float[] mult = new float[arr.length];
        for(int i = 0; i < arr.length; i++){
            mult[i] = c * arr[i];
        }
        return mult;
    }

    public static float[] add(float[] arr, float c){
        float[] mult = new float[arr.length];
        for(int i = 0; i < arr.length; i++){
            mult[i] = c + arr[i];
        }
        return mult;
    }

    public static float[] add(float[] a, float[] b){
        float[] mult = new float[a.length];
        for(int i = 0; i < a.length; i++){
            mult[i] = b[i] + a[i];
        }
        return mult;
    }


    public static Matrix mean(Matrix m, int axis){
        ArrayList<Integer> size = new ArrayList<>();
        size.add(0);
        size.add(1);
        size.add(2);
        int ia = size.remove(axis);
        int ka = size.get(0);
        int ja = size.get(1);
        Matrix transpose = m.transpose(ka, ja, ia);
        Tensor[][] mean = new Tensor[transpose.depth][transpose.rows];
        for(int i = 0; i < mean[0].length; i++){
            for(int j = 0; j < mean.length; j++){
                mean[j][i] = mean(transpose.contents3d[j][i]);
            }
        }
        return new Matrix(mean);
    }

    public static Tensor[] sum(Matrix m, int axis){
        Matrix transpose = m;
        Tensor[] sum = new Tensor[m.shape(axis)];
        if(axis == 1){
            transpose = m.T();
        }
        for(int i = 0; i < transpose.rows; i++){
            sum[i] = sum(transpose.contents[i]);
        }
        return sum;
    }

    public static Tensor sum(Tensor[] arr){
        double total = 0;
        for(Tensor f : arr){
            total += f.getData();
        }
        return Tensor.param(total);
    }

    public static float constrain(float value, float min, float max) {
        return Math.min(max, Math.max(min, value));
    }


    public static float randomGauss() {
        return (float) random.nextGaussian();
    }

    public static float random() {
        return random.nextFloat();
    }


    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    public static void println(Object out) {
        System.out.println(out);
    }

    public static void print(Object out) {
        System.out.print(out);
    }

    public static void printArray(float[] arr) {
        print('[');
        for (int i = 0; i < arr.length; i++) {
            print(arr[i] + (i < arr.length - 1 ? ", " : ""));
        }
        println(']');
    }

    public static void printArray(Object[] arr) {
        print('[');
        for (int i = 0; i < arr.length; i++) {
            print(arr[i] + (i < arr.length - 1 ? ", " : ""));
        }
        println(']');
    }

    public static void printArray(String[] arr) {
        print('[');
        for (int i = 0; i < arr.length; i++) {
            print(arr[i] + (i < arr.length - 1 ? ", " : ""));
        }
        println(']');
    }

    public static void printArray(float[][] arr) {
        print('[');
        for (int i = 0; i < arr.length; i++) {
            print('[');
            for (int j = 0; j < arr[i].length; j++) {
                print(arr[i][j] + (j < arr[i].length - 1 ? " " : ""));
            }
            println(']' + (i < arr.length - 1 ? " " : ""));
        }
        println(']');
    }


}
