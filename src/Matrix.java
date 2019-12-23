import java.util.concurrent.atomic.AtomicInteger;
import static java.lang.System.out;

public class Matrix  {
    public Tensor[][] contents;
    public Tensor[][][] contents3d;
    public int columns;
    public int rows;
    public int depth;
    private Matrix T;
    boolean is3d = false;

    public Matrix(Tensor[][] contents) {
        this.contents = contents;
        rows = this.contents.length;
        columns = this.contents[0].length;

    }

    public Matrix(Tensor[][][] contents3d) {
        this.contents3d = contents3d;
        depth = contents3d.length;
        rows = contents3d[0].length;
        columns = contents3d[0][0].length;
        is3d = true;
    }

    public Matrix(int rows, int columns) {
        this(new Tensor[rows][columns]);
    }

    public double mean(){
        double total = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                total += contents[i][j].getData();
            }
        }
        return total/(rows * columns);
    }

    public static Matrix rand(int rows, int columns) {
        Tensor[][] rand = new Tensor[rows][columns];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                rand[j][i] = Tensor.variable(Math.random() * 2 - 1);
            }
        }
        return new Matrix(rand);
    }

    public static Matrix zeros(int rows, int columns) {
        Tensor[][] zeros = new Tensor[rows][columns];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                zeros[j][i] = Tensor.variable(0);
            }
        }
        return new Matrix(zeros);
    }

    public Matrix addSingle(Matrix b){
        Tensor[][] add = new Tensor[rows][columns];
        Tensor[][] ca = contents;
        Tensor[][] cb = b.contents;

        for(int i = 0; i < columns; i++){
            for(int j = 0; j < rows; j++){
                add[j][i] = Tensor.plus(ca[j][i], cb[j][0]);
            }
        }
        return new Matrix(add);
    }

    public static Matrix ones(int rows, int columns) {
        Matrix matrix = new Matrix(rows, columns);

        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                matrix.contents[j][i] = Tensor.variable(0);
            }
        }
        return matrix;
    }


    public Matrix T() {
        if (T != null) return T;
        Tensor[][] transpose = new Tensor[columns][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                transpose[j][i] = contents[i][j];
            }
        }
        T = new Matrix(transpose);
        return T;
    }

    public Matrix transpose(int ka, int ia, int ja) {
        int[] size = new int[]{depth, rows, columns};
        Tensor[][][] transpose = new Tensor[size[ka]][size[ia]][size[ja]];
        AtomicInteger i = new AtomicInteger();
        AtomicInteger j = new AtomicInteger();
        AtomicInteger k = new AtomicInteger();
        AtomicInteger[] d = new AtomicInteger[]{k, j, i};

        for (k.set(0); k.get() < depth; k.set(k.get() + 1)) {
            for (i.set(0); i.get() < columns; i.set(i.get() + 1)) {
                for (j.set(0); j.get() < rows; j.set(j.get() + 1)) {
                    transpose[d[ka].get()][d[ia].get()][d[ja].get()] = contents3d[k.get()][j.get()][i.get()];
                }
            }
        }
        return new Matrix(transpose);
    }


    public Matrix to3d() {
        Tensor[][][] contents3d = new Tensor[1][rows][columns];
        for (int i = 0; i < rows; i++) {
            contents3d[0][i] = contents[i].clone();
        }
        return new Matrix(contents3d);
    }


    public Matrix sub(Matrix b) {
        Tensor[][] ca = contents;
        Tensor[][] cb = b.contents;
        Tensor[][] sub = new Tensor[rows][columns];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                sub[j][i] = Tensor.minus(ca[j][i], cb[j][i]);
            }
        }
        return new Matrix(sub);
    }

    public int shape(int index) {
        return index == 0 ? rows : columns;
    }

    public Matrix dot(Matrix b) {
        Tensor[][] ca = contents;
        Tensor[][] cb = b.contents;
        Tensor[][] dot = Matrix.zeros(rows, b.columns).contents;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < b.columns; j++) {
                for (int p = 0; p < b.rows; p++) {
                    dot[i][j] = Tensor.plus(Tensor.multiply(ca[i][p], cb[p][j]), dot[i][j]);
                }
            }
        }
        return new Matrix(dot);
    }

    public Matrix multiply(Matrix b) {
        Tensor[][] mult = new Tensor[rows][columns];
        Tensor[][] ca = contents;
        Tensor[][] cb = b.contents;
        if (b.columns == 1 && b.rows == 1) {
            for (int i = 0; i < columns; i++) {
                for (int j = 0; j < rows; j++) {
                    mult[j][i] = Tensor.multiply(ca[j][i], cb[0][0]);
                }
            }
        } else {
            for (int i = 0; i < columns; i++) {
                for (int j = 0; j < rows; j++) {
                    mult[j][i] = Tensor.multiply(ca[j][i], cb[j][i]);
                }
            }
        }
        return new Matrix(mult);
    }

    public Matrix multiply(Tensor c){
        Tensor[][] mult = new Tensor[rows][columns];
        for(int i = 0; i < columns; i++){
            for(int j = 0; j < rows; j++){
                mult[j][i] = Tensor.multiply(c, contents[j][i]);
            }
        }
        return new Matrix(mult);
    }

    public Matrix add(Matrix b) {
        Tensor[][] add = new Tensor[rows][columns];
        Tensor[][] ca = contents;
        Tensor[][] cb = b.contents;
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows; j++) {
                add[j][i] = Tensor.plus(ca[j][i], cb[j][i]);
            }
        }
        return new Matrix(add);
    }


    public Matrix cross3d(Matrix b) {
        Tensor[][][] catomic = contents3d;
        Tensor[][][] cmatrix = b.contents3d;
        Tensor[][][] cmult = new Tensor[b.depth][b.rows][b.columns];

        for (int i = 0; i < b.depth; i++) {
            Matrix atomic = new Matrix(catomic[0]);
            Matrix matrix = new Matrix(cmatrix[i]);
            cmult[i] = atomic.cross(matrix).contents;
        }
        return new Matrix(cmult);
    }

    public Matrix cross(Matrix b) {
        int rows = Math.max(this.rows, b.rows);
        int columns = Math.max(this.columns, b.columns);
        Tensor[][] cross = new Tensor[rows][columns];
        for(int i = 0; i < columns; i++){
            for(int j = 0; j < rows; j++){
                if(this.rows == 1) {
                    cross[j][i] = Tensor.multiply(contents[0][i], b.contents[j][0]);
                }else if(columns == 1){
                    cross[j][i] = Tensor.multiply(contents[j][0], b.contents[0][i]);
                }
            }
        }

        return new Matrix(cross);
    }

    public Matrix removeBottom() {
        Tensor[][] rem = new Tensor[rows - 1][columns];
        for (int i = 0; i < columns; i++) {
            for (int j = 0; j < rows - 1; j++) {
                rem[j][i] = contents[j][i];
            }
        }
        return new Matrix(rem);
    }

    public static Matrix[] arrayCopy(Matrix[] src){
        Matrix[] dest = new Matrix[src.length];
        for(int i = 0; i < src.length; i++){
            dest[i] = src[i] != null ? src[i].copy() : null;
        }
        return dest;
    }

    public Matrix copy(){
        Tensor[][] copy = new Tensor[rows][columns];
        for(int i = 0; i< columns; i++){
            for(int j = 0; j < rows; j++){
                copy[j][i] = contents[j][i];
            }
        }
        return new Matrix(copy);
    }


    public void print() {
        Tensor[][] arr = contents;
        out.print('[');
        for (int i = 0; i < arr.length; i++) {
            out.print(i > 0 ? " [" : '[');
            for (int j = 0; j < arr[i].length; j++) {
                String out = String.format(arr[i][j].getData() < 0 ? "%6f" : " %6f", arr[i][j].getData());
                System.out.print(out + (j < arr[i].length - 1 ? " " : ""));
            }
            out.print(']' + (i < arr.length - 1 ? "\n" : ""));
        }
        out.println("]\n");
    }

    public void print3d() {
        Tensor[][][] arr = contents3d;
        out.print('[');
        for (int k = 0; k < depth; k++) {
            out.print('[');
            for (int i = 0; i < arr[k].length; i++) {
                out.print(i > 0 ? " [" : '[');
                for (int j = 0; j < arr[k][i].length; j++) {
                    String out = String.format(arr[k][i][j].getData() < 0 ? "%6f" : " %6f", arr[k][i][j]);
                    System.out.print(out + (j < arr[k][i].length - 1 ? " " : ""));
                }
                out.print(']' + (i < arr[k].length - 1 ? "\n" : ""));
            }
            out.println("]");
        }
        out.println(']');
    }

    public String[] toStringArray(){
        String[] array = new String[columns*rows];
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                int index = j + i * columns;
                array[index] = String.valueOf(contents[i][j].getData());
            }
        }
        return array;
    }

    public Matrix sq(){
        Matrix sq = new Matrix(rows, columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sq.contents[i][j] = Tensor.sq(contents[i][j]);
            }
        }

        return sq;
    }

    public void backward(){

        for(int i = 0; i < contents.length; i++) {
            for(int j = 0; j < contents[0].length; j++){
                contents[i][j].backward();
            }
        }
    }


    public void toMatrix(String[] array){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                int index = j + i * columns;
                float val = Float.parseFloat(array[index]);
                contents[i][j] = Tensor.variable(val);
            }
        }
    }

    public static Matrix vstack(Matrix top, Matrix bottom) {
        int rowsSum = top.rows + bottom.rows;
        Tensor[][] stack = new Tensor[rowsSum][top.columns];

        for (int i = 0; i < top.columns; i++) {
            for (int j = 0; j < top.rows; j++) {
                stack[j][i] = top.contents[j][i];
            }
        }

        for (int i = 0; i < bottom.columns; i++) {
            for (int j = 0; j < bottom.rows; j++) {
                stack[j + top.rows][i] = bottom.contents[j][i];
            }
        }
        return new Matrix(stack);
    }

    public static Matrix hstack(Matrix left, Matrix right) {
        int colssum = left.columns + right.columns;
        Tensor[][] stack = new Tensor[left.rows][colssum];

        for (int i = 0; i < left.columns; i++) {
            for (int j = 0; j < left.rows; j++) {
                stack[j][i] = left.contents[j][i];
            }
        }

        for (int i = 0; i < right.columns; i++) {
            for (int j = 0; j < right.rows; j++) {
                stack[j][i + left.columns] = right.contents[j][i];
            }
        }
        return new Matrix(stack);
    }

}
