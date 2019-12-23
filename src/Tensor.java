import java.util.function.Consumer;

/**
 * @author Sergey Kuptsov
 */
public class Tensor {
    /**
     * actual data
     */
    double data;
    /**
     * derivative value
     */
    double derivative = 0;
    /**
     * function for calculating derivative
     */
    private Consumer<Double> autoDifferentiationFn;
    /**
     * is requires derivative
     */
    private boolean requiresDerivative;

    private Tensor(double data, boolean requiresDerivative) {
        this.data = data;
        this.requiresDerivative = requiresDerivative;
        this.autoDifferentiationFn = (prevDerivative) -> derivative += prevDerivative;
    }

    public static Tensor variable(double data) {
        return new Tensor(data, true);
    }

    private static final Consumer<Double> emptyFn = v -> {
    };

    private Tensor(double data, boolean requiresDerivative, Consumer<Double> autoDifferentiationFn) {
        this.data = data;
        this.requiresDerivative = requiresDerivative;
        this.autoDifferentiationFn = autoDifferentiationFn;
    }

    public static Tensor param(double data) {
        return new Tensor(data, false, emptyFn);
    }

    public static Tensor multiply(Tensor tensor1, Tensor tensor2) {
        double data = tensor1.data * tensor2.data;
        boolean requiresDerivative = tensor1.requiresDerivative || tensor2.requiresDerivative;

        Consumer<Double> node1F = tensor1.requiresDerivative ?
                prevGrad -> tensor1.autoDifferentiationFn
                        .accept(prevGrad * tensor2.data)
                : emptyFn;

        Consumer<Double> node2F = tensor2.requiresDerivative ?
                prevGrad -> tensor2.autoDifferentiationFn
                        .accept(prevGrad * tensor1.data)
                : emptyFn;

        Consumer<Double> autoDifferentiationFn = v -> {
            node1F.accept(v);
            node2F.accept(v);
        };

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor plus(Tensor tensor1, Tensor tensor2) {
        boolean requiresDerivative = tensor1.requiresDerivative || tensor2.requiresDerivative;
        double data = tensor1.data + tensor2.data;

        Consumer<Double> node1F = tensor1.requiresDerivative ?
                prevGrad -> tensor1.autoDifferentiationFn
                        .accept(prevGrad)
                : emptyFn;

        Consumer<Double> node2F = tensor2.requiresDerivative ?
                prevGrad -> tensor2.autoDifferentiationFn
                        .accept(prevGrad)
                : emptyFn;

        Consumer<Double> autoDifferentiationFn = prevGrad -> {
            node1F.accept(prevGrad);
            node2F.accept(prevGrad);
        };

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor minus(Tensor tensor1, Tensor tensor2) {
        boolean requiresDerivative = tensor1.requiresDerivative || tensor2.requiresDerivative;
        double data = tensor1.data - tensor2.data;

        Consumer<Double> node1F = tensor1.requiresDerivative ?
                prevGrad -> tensor1.autoDifferentiationFn
                        .accept(prevGrad)
                : emptyFn;

        Consumer<Double> node2F = tensor2.requiresDerivative ?
                prevGrad -> tensor2.autoDifferentiationFn
                        .accept(prevGrad)
                : emptyFn;

        Consumer<Double> autoDifferentiationFn = prevGrad -> {
            node1F.accept(prevGrad);
            node2F.accept(prevGrad);
        };

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor sq(Tensor tensor) {
        double data = tensor.data * tensor.data;
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad * 2 * tensor.data)
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor relu(Tensor tensor){
        double data = Math.max(tensor.data, 0);
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad * (tensor.data < 0 ? 0 : 1))
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor leakyRelu(Tensor tensor){
        double data = tensor.data < 0 ? 0.01 * tensor.data : tensor.data;
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad * (tensor.data < 0 ? 0.01 : 1))
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor linear(Tensor tensor){
        double data = tensor.data;
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad)
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor tanh(Tensor tensor){
        double data = Math.tanh(tensor.data);
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad * (1 - data*data))
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }

    public static Tensor sigmoid(Tensor tensor){
        double data = 1/(1 + Math.exp(-tensor.data));
        boolean requiresDerivative = tensor.requiresDerivative;

        Consumer<Double> autoDifferentiationFn = tensor.requiresDerivative
                ? prevGrad -> tensor.autoDifferentiationFn.accept(prevGrad * (data * (1 - data)))
                : emptyFn;

        return new Tensor(data, requiresDerivative, autoDifferentiationFn);
    }



    public double grad() {
        return derivative;
    }

    public double getData() {
        return data;
    }

    public void backward() {
        autoDifferentiationFn.accept(1.0);
    }
}