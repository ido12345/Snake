#ifndef ML_H
#define ML_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>
#endif

typedef enum ACTIVATION_TYPES
{
    SIGMOID,
    RELU,
    LEAKYRELU,
} ActivationType;

typedef struct Activation
{
    ActivationType type;
    float (*activationFunc)(float);
} Activation;

typedef struct Matrix
{
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Matrix;

typedef struct Step
{
    Matrix state;
    float reward;
    int action;
    float probability;
} Step;

typedef struct Network
{
    Matrix *layers;
    Matrix *weights;
    Matrix *biases;
    Activation *activations;
    size_t count;
} Network;

#define ARR_LEN(arr) (sizeof(arr) / sizeof(*(arr)))

#define MAT_AT(M, i, j) ((M).es[(i) * (M).stride + (j)])

#define PRINT_MAT(m) print_mat((m), #m, 0, "%f")
#define PRINT_NETWORK(nn) print_Network((nn), #nn, false)
#define NETWORK_IN(nn) ((nn).layers[0])
#define NETWORK_OUT(nn) ((nn).layers[(nn).count])

#define SOFTMAX_OUTPUTS(nn) (softmaxf(NETWORK_OUT(nn).es, NETWORK_OUT(nn).es, NETWORK_OUT(nn).cols))

float rand_float();
float sigmoidf(float x);
float reluf(float x);
float leakyreluf(float x);
float sigmoidDerivative(float x);
float reluDerivative(float x);
void softmaxf(float *dest, float *x, int n);
float (*getActFunc(ActivationType a))(float);
char *getActName(ActivationType a);
float (*getActDerivative(ActivationType a))(float);

// float safe_expf(float x)
// {
//     if (x > 88.72f)
//         return expf(88.72f); // Prevent overflow
//     if (x < -88.72f)
//         return 0.0f; // Underflow case
//     return expf(x);
// }

float sigmoidf(float x)
{
    return (1.f / (1.f + expf(-x)));
}

float reluf(float x)
{
    return (x > 0.f ? x : 0.f);
}

float leakyreluf(float x)
{
    return (x > 0.f ? 0.01f * x : 0.f);
}

float sigmoidDerivative(float x)
{
    return (x) * (1 - x);
}
float reluDerivative(float x)
{
    return (x > 0.f ? 1.f : 0.f);
}

float leakyreluDerivative(float x)
{
    return (x > 0.f ? 0.01f : 0.f);
}

void softmaxf(float *dest, float *x, int n)
{
    // Step 1: Find the maximum value in x
    float max_x = x[0];
    for (int i = 1; i < n; i++)
    {
        if (x[i] > max_x)
        {
            max_x = x[i];
        }
    }

    // Step 2: Compute exponentials and their sum
    float sum = 0.f;
    for (int i = 0; i < n; i++)
    {
        dest[i] = exp(x[i] - max_x); // Subtract max_x for numerical stability
        sum += dest[i];
    }

    // Step 3: Normalize to get probabilities
    for (int i = 0; i < n; i++)
    {
        dest[i] /= sum;
    }
}

float (*getActFunc(ActivationType a))(float)
{
    switch (a)
    {
    case SIGMOID:
        return sigmoidf;
    case RELU:
        return reluf;
    case LEAKYRELU:
        return leakyreluf;
    default:
        return NULL;
    }
}

char *getActName(ActivationType a)
{
    switch (a)
    {
    case SIGMOID:
        return "Sigmoid";
    case RELU:
        return "ReLU";
    case LEAKYRELU:
        return "LeakyReLU";
    default:
        return NULL;
    }
}

float (*getActDerivative(ActivationType a))(float)
{
    switch (a)
    {
    case SIGMOID:
        return sigmoidDerivative;
    case RELU:
        return reluDerivative;
    case LEAKYRELU:
        return leakyreluDerivative;
    default:
        return NULL;
    }
}

float rand_float()
{
    return ((float)rand() / (float)RAND_MAX);
}

Matrix mat_alloc(size_t rows, size_t cols);
void mat_dot(Matrix dest, Matrix a, Matrix b);
void mat_sum(Matrix dest, Matrix src);
void mat_activate(Matrix m, float (*actFunc)(float));
void mat_sig(Matrix m);
void mat_rand(Matrix m, float low, float high);
Matrix mat_row(Matrix src, size_t row);
Matrix mat_col(Matrix src, size_t col);
void mat_copy(Matrix dest, Matrix src);
void mat_clear(Matrix m);
void print_mat(Matrix m, const char *name, int padding, const char *format);
void print_activation(Activation a, const char *name, int padding);
bool mat_same(Matrix a, Matrix b);
void fwrite_mat(Matrix m, FILE *dest);
void fread_mat(Matrix m, FILE *src);
void shuffle_mat(Matrix m);

Network NeuralNetwork(size_t *layers, size_t count, ActivationType *activations);
void print_Network(Network nn, const char *name, bool showLayers);
void Network_rand(Network nn, float low, float high);
float Network_cost(Network nn, Matrix in, Matrix out);
float Network_cross_entropy_cost(Network nn, Step *steps[], size_t stepAmount);
void Network_forward(Network nn);
void Network_diff(Network nn, Network g, float eps, Matrix in, Matrix out);
void Network_policy_gradient_diff(Network nn, Network g, float eps, Step *steps[], size_t stepAmount);
void Network_backprop(Network nn, Network g, Matrix in, Matrix out);
void Network_policy_gradient_backprop(Network nn, Network g, Step *steps[], size_t stepAmount);
void Network_clear(Network nn);
void Network_learn(Network nn, Network g, float rate);
bool Network_same(Network a, Network b);
void Network_save(Network nn, const char *fileName);
void Network_load(Network nn, const char *fileName);
size_t *Network_getArch(Network nn);
bool Network_cmpArch(Network nn, size_t *arch, size_t archLen);

const char fileExtension[] = ".netw";
const char fileHeader[] = "nn";
const char fileMatRow = '\n';

void mat_shuffle_rows(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        size_t j = (i + rand() % (m.rows - i));
        if (i == j)
            continue;
        for (size_t k = 0; k < m.cols; k++)
        {
            float temp = MAT_AT(m, i, k);
            MAT_AT(m, i, k) = MAT_AT(m, j, k);
            MAT_AT(m, j, k) = temp;
        }
    }
}

size_t *Network_getArch(Network nn)
{
    size_t *arch = malloc(sizeof(*arch) * (nn.count + 1));
    for (size_t i = 0; i < nn.count; i++)
    {
        arch[i] = nn.weights[i].rows;
    }
    arch[nn.count] = NETWORK_OUT(nn).rows;
    return arch;
}

bool Network_cmpArch(Network nn, size_t *arch, size_t archLen)
{
    if (nn.count + 1 != archLen)
        return false;

    for (size_t i = 0; i < nn.count; i++)
    {
        if (arch[i] != nn.weights[i].rows)
            return false;
    }
    if (arch[nn.count] != NETWORK_OUT(nn).rows)
        return false;
    return true;
}

void Network_save(Network nn, const char *fileName)
{
#if defined(_WIN32) || defined(_WIN64)
    char path[MAX_PATH];
    unsigned long length = GetModuleFileName(NULL, path, sizeof(path));
    if (!length)
    {
        fprintf(stderr, "Failed to get file path\n");
        return;
    }
    unsigned long writeCounter = 0;
    for (unsigned long i = length - 1; i >= 0; i--)
    {
        if (path[i - 1] == '\\')
        {
            path[i] = '\0';
            writeCounter = i;
            break;
        }
    }
    strcat(path, fileName);
    strcat(path, fileExtension);

    FILE *networkFile = fopen(path, "r");
    if (networkFile)
    {
        fprintf(stderr, "File already exists\n");
        return;
    }
    networkFile = fopen(path, "wb");
    if (!networkFile)
    {
        fprintf(stderr, "File could not be opened\n");
        return;
    }
#endif

    // Writing the file
    fwrite(fileHeader, sizeof(char), sizeof(fileHeader) - 1, networkFile);
    size_t *arch = Network_getArch(nn);
    size_t archLen = nn.count + 1;
    fwrite(&archLen, sizeof(archLen), 1, networkFile);
    fwrite(arch, sizeof(*arch), nn.count + 1, networkFile);
    for (size_t i = 0; i < nn.count; i++)
    {
        fwrite_mat(nn.weights[i], networkFile);
        fwrite_mat(nn.biases[i], networkFile);
    }
    fclose(networkFile);
    printf("File saved successfully\n");
}

void Network_load(Network nn, const char *fileName)
{
#if defined(_WIN32) || defined(_WIN64)
    char path[MAX_PATH];
    unsigned long length = GetModuleFileName(NULL, path, sizeof(path));
    if (!length)
    {
        fprintf(stderr, "Failed to get file path\n");
        return;
    }
    unsigned long writeCounter = 0;
    for (unsigned long i = length - 1; i >= 0; i--)
    {
        if (path[i - 1] == '\\')
        {
            path[i] = '\0';
            writeCounter = i;
            break;
        }
    }
    strcat(path, fileName);
    strcat(path, fileExtension);

    FILE *networkFile = fopen(path, "rb");
    if (!networkFile)
    {
        fprintf(stderr, "File could not be opened\n");
        return;
    }
#endif

    // Reading the file
    unsigned long headerLen = sizeof(fileHeader) - 1;
    char header[sizeof(fileHeader) - 1];
    fread(header, sizeof(*fileHeader), headerLen, networkFile);
    if (strncmp(header, fileHeader, headerLen) != 0)
    {
        fprintf(stderr, "Invalid %s file\n", fileExtension);
        fclose(networkFile);
        return;
    }
    size_t archLen;
    fread(&archLen, sizeof(archLen), 1, networkFile);
    size_t *arch = (size_t *)malloc(sizeof(*arch) * archLen);
    fread(arch, sizeof(*arch), archLen, networkFile);

    if (!Network_cmpArch(nn, arch, archLen))
    {
        fprintf(stderr, "Provided Network architecture is not the same as loaded Network\n");
        fclose(networkFile);
        return;
    }
    for (size_t i = 0; i < nn.count; i++)
    {
        fread_mat(nn.weights[i], networkFile);
        fread_mat(nn.biases[i], networkFile);
    }
    fclose(networkFile);
    printf("File loaded successfully\n");
}

void fwrite_mat(Matrix src, FILE *dest)
{
    for (size_t i = 0; i < src.rows; i++)
    {
        fwrite(&MAT_AT(src, i, 0), sizeof(*src.es), src.cols, dest);
        fwrite(&fileMatRow, sizeof(fileMatRow), 1, dest); // could be removed
    }
}

void fread_mat(Matrix dest, FILE *src)
{
    for (size_t i = 0; i < dest.rows; i++)
    {
        fread(&MAT_AT(dest, i, 0), sizeof(*dest.es), dest.cols, src);
        char temp;                                // could be removed
        fread(&temp, sizeof(fileMatRow), 1, src); // could be removed
    }
}

void print_mat(Matrix m, const char *name, int padding, const char *format)
{
    printf("%*s%s = [\n", padding, "", name);
    for (size_t i = 0; i < m.rows; i++)
    {
        printf("%*s    ", padding, "");
        for (size_t j = 0; j < m.cols; j++)
        {
            printf(format, MAT_AT(m, i, j));
            printf("  ");
        }
        printf("\n");
    }
    printf("%*s]\n", padding, "");
}

void print_activation(Activation a, const char *name, int padding)
{
    char *actName = getActName(a.type);
    printf("%*s%s = %s\n", padding, "", name, actName);
}

void mat_dot(Matrix dest, Matrix a, Matrix b)
{
    if (a.cols != b.rows)
        return;
    if (dest.rows != a.rows)
        return;
    if (dest.cols != b.cols)
        return;
    size_t n = a.cols;

    mat_clear(dest);
    for (size_t i = 0; i < dest.rows; i++)
    {
        for (size_t j = 0; j < dest.cols; j++)
        {
            for (size_t k = 0; k < n; k++)
            {
                MAT_AT(dest, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Matrix dest, Matrix src)
{
    if (!mat_same(dest, src))
        return;

    for (size_t i = 0; i < dest.rows; i++)
    {
        for (size_t j = 0; j < dest.cols; j++)
        {
            MAT_AT(dest, i, j) += MAT_AT(src, i, j);
        }
    }
}

void mat_sig(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_activate(Matrix m, float (*actFunc)(float))
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = actFunc(MAT_AT(m, i, j));
        }
    }
}

void mat_rand(Matrix m, float low, float high)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

Matrix mat_row(Matrix src, size_t row)
{
    Matrix m;
    m.rows = 1;
    m.cols = src.cols;
    m.stride = src.stride;
    m.es = &MAT_AT(src, row, 0);
    return m;
}

Matrix mat_col(Matrix src, size_t col)
{
    Matrix m;
    m.rows = src.rows;
    m.cols = 1;
    m.stride = src.stride;
    m.es = &MAT_AT(src, 0, col);
    return m;
}

void mat_copy(Matrix dest, Matrix src)
{
    if (!mat_same(dest, src))
        return;

    for (size_t i = 0; i < dest.rows; i++)
    {
        for (size_t j = 0; j < dest.cols; j++)
        {
            MAT_AT(dest, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_clear(Matrix m)
{
    for (size_t i = 0; i < m.rows; i++)
    {
        for (size_t j = 0; j < m.cols; j++)
        {
            MAT_AT(m, i, j) = 0;
        }
    }
}

Matrix mat_alloc(size_t rows, size_t cols)
{
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = calloc(sizeof(m.es) * rows * cols, 1);
    return m;
}

bool mat_same(Matrix a, Matrix b)
{
    return ((a.rows == b.rows) && (a.cols == b.cols));
}

bool Network_same(Network a, Network b)
{
    if (a.count != b.count)
        return false;

    for (size_t i = 0; i < a.count; i++)
    {
        if (!mat_same(a.layers[i], b.layers[i]))
            return false;
        if (!mat_same(a.weights[i], b.weights[i]))
            return false;
        if (!mat_same(a.biases[i], b.biases[i]))
            return false;
    }
    return true;
}

Network NeuralNetwork(size_t *layers, size_t count, ActivationType *activations)
{
    Network nn;
    nn.count = count - 1;
    nn.layers = malloc(sizeof(*nn.layers) * nn.count + 1);
    nn.weights = malloc(sizeof(*nn.weights) * nn.count);
    nn.biases = malloc(sizeof(*nn.biases) * nn.count);
    if (activations != NULL)
    {
        nn.activations = malloc(sizeof(*nn.activations) * nn.count);
    }
    else
    {
        nn.activations = NULL;
    }

    nn.layers[0] = mat_alloc(1, layers[0]);
    for (size_t i = 0; i < nn.count; i++)
    {
        nn.weights[i] = mat_alloc(layers[i], layers[i + 1]);
        nn.biases[i] = mat_alloc(1, layers[i + 1]);
        if (activations != NULL)
        {
            nn.activations[i].type = activations[i];
            nn.activations[i].activationFunc = getActFunc(activations[i]);
        }
        nn.layers[i + 1] = mat_alloc(1, layers[i + 1]);
    }
    return nn;
}

void print_Network(Network nn, const char *name, bool showLayers)
{
    char buff[100];
    printf("%s = [\n", name);
    for (size_t i = 0; i < nn.count; i++)
    {
        if (showLayers)
        {
            snprintf(buff, sizeof(buff), "%s.layers[%zu]", name, i);
            print_mat(nn.layers[i], buff, 4, "%f");
        }
        snprintf(buff, sizeof(buff), "%s.weights[%zu]", name, i);
        print_mat(nn.weights[i], buff, 4, "%f");
        snprintf(buff, sizeof(buff), "%s.biases[%zu]", name, i);
        print_mat(nn.biases[i], buff, 4, "%f");
        if (nn.activations)
        {
            snprintf(buff, sizeof(buff), "%s.activations[%zu]", name, i);
            print_activation(nn.activations[i], buff, 4);
        }
    }
    if (showLayers)
    {
        snprintf(buff, sizeof(buff), "%s.layers[%zu]", name, nn.count);
        print_mat(nn.layers[nn.count], buff, 4, "%f");
    }
    printf("]\n");
}

void Network_rand(Network nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        // mat_rand(nn.layers[i], low, high);
        mat_rand(nn.weights[i], low, high);
        mat_rand(nn.biases[i], low, high);
    }
}

void Network_clear(Network nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_clear(nn.layers[i]);
        mat_clear(nn.weights[i]);
        mat_clear(nn.biases[i]);
    }
    mat_clear(nn.layers[nn.count]);
}

float Network_cost(Network nn, Matrix in, Matrix out)
{
    if (NETWORK_IN(nn).cols != in.cols)
        return -1.f;
    if (NETWORK_OUT(nn).cols != in.cols)
        return -1.f;

    float result = 0.f;
    for (size_t i = 0; i < in.rows; i++)
    {
        Matrix in_row = mat_row(in, i);
        mat_copy(NETWORK_IN(nn), in_row);
        Network_forward(nn);

        for (size_t j = 0; j < out.cols; j++)
        {
            float d = MAT_AT(NETWORK_OUT(nn), 0, j) - MAT_AT(out, i, j);
            result += d * d;
        }
    }

    return result / in.rows;
}

float Network_cross_entropy_cost(Network nn, Step *steps[], size_t stepAmount)
{
    float cost = 0.f;
    for (size_t i = 0; i < stepAmount; i++)
    {
        Step temp = *steps[i];
        if (steps[i]->reward != 0)
        {
            float logP = -log(steps[i]->probability);
            cost += steps[i]->reward * logP;
        }
    }
    return cost;
}

void Network_forward(Network nn)
{
    for (size_t i = 0; i < nn.count; i++)
    {
        mat_dot(nn.layers[i + 1], nn.layers[i], nn.weights[i]);
        mat_sum(nn.layers[i + 1], nn.biases[i]);
        if (nn.activations)
        {
            mat_activate(nn.layers[i + 1], nn.activations[i].activationFunc);
        }
    }
}

void Network_diff(Network nn, Network g, float eps, Matrix in, Matrix out)
{
    if (in.rows != out.rows)
        return;
    if (!mat_same(NETWORK_IN(nn), mat_row(in, 0)))
        return;
    if (!mat_same(NETWORK_OUT(nn), mat_row(out, 0)))
        return;
    if (!Network_same(nn, g))
        return;

    float saved;
    float cost = Network_cost(nn, in, out);
    for (size_t i = 0; i < nn.count; i++)
    {
        Matrix weights = nn.weights[i];
        for (size_t j = 0; j < weights.rows; j++)
        {
            for (size_t k = 0; k < weights.cols; k++)
            {
                saved = MAT_AT(weights, j, k);
                MAT_AT(weights, j, k) += eps;
                float newCost = Network_cost(nn, in, out);
                MAT_AT(g.weights[i], j, k) = (newCost - cost) / eps;
                MAT_AT(weights, j, k) = saved;
            }
        }

        Matrix biases = nn.biases[i];
        for (size_t j = 0; j < biases.rows; j++)
        {
            for (size_t k = 0; k < biases.cols; k++)
            {
                saved = MAT_AT(biases, j, k);
                MAT_AT(biases, j, k) += eps;
                float newCost = Network_cost(nn, in, out);
                MAT_AT(g.biases[i], j, k) = (newCost - cost) / eps;
                MAT_AT(biases, j, k) = saved;
            }
        }
    }
}

void Network_policy_gradient_diff(Network nn, Network g, float eps, Step *steps[], size_t stepAmount)
{
    if (NETWORK_IN(nn).cols != steps[0]->state.cols)
        return;
    if (!Network_same(nn, g))
        return;

    float saved;
    float cost = Network_cross_entropy_cost(nn, steps, stepAmount);
    for (size_t i = 0; i < nn.count; i++)
    {
        Matrix weights = nn.weights[i];
        for (size_t j = 0; j < weights.rows; j++)
        {
            for (size_t k = 0; k < weights.cols; k++)
            {
                saved = MAT_AT(weights, j, k);
                MAT_AT(weights, j, k) += eps;
                float newCost = Network_cross_entropy_cost(nn, steps, stepAmount);
                MAT_AT(g.weights[i], j, k) = (newCost - cost) / eps;
                MAT_AT(weights, j, k) = saved;
            }
        }

        Matrix biases = nn.biases[i];
        for (size_t k = 0; k < biases.cols; k++)
        {
            saved = MAT_AT(biases, 0, k);
            MAT_AT(biases, 0, k) += eps;
            float newCost = Network_cross_entropy_cost(nn, steps, stepAmount);
            MAT_AT(g.biases[i], 0, k) = (newCost - cost) / eps;
            MAT_AT(biases, 0, k) = saved;
        }
    }
}

void Network_backprop(Network nn, Network g, Matrix in, Matrix out)
{
    if (in.rows != out.rows)
        return;
    if (!mat_same(NETWORK_IN(nn), mat_row(in, 0)))
        return;
    if (!mat_same(NETWORK_OUT(nn), mat_row(out, 0)))
        return;
    if (!Network_same(nn, g))
        return;
    size_t n = in.rows; // amount of samples

    Network_clear(g);

    // i = current sample
    // l = current layer
    // j = current "node"
    // k = previous "node"

    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NETWORK_IN(nn), mat_row(in, i));
        Network_forward(nn);

        for (size_t j = 0; j <= g.count; j++)
        {
            mat_clear(g.layers[j]);
        }

        for (size_t j = 0; j < out.cols; j++)
        {
#ifdef TRAD_BACKPROP
            MAT_AT(NETWORK_OUT(g), 0, j) = 2 * (MAT_AT(NETWORK_OUT(nn), 0, j) - MAT_AT(out, i, j));
#else
            MAT_AT(NETWORK_OUT(g), 0, j) = (MAT_AT(NETWORK_OUT(nn), 0, j) - MAT_AT(out, 0, j));
#endif // TRAD_BACKPROP
        }

#ifdef TRAD_BACKPROP
        float s = 1.f;
#else
        float s = 2.f;
#endif // TRAD_BACKPROP

        for (size_t l = nn.count; l > 0; l--)
        {
            for (size_t j = 0; j < nn.layers[l].cols; j++)
            {
                float outputAhead = MAT_AT(nn.layers[l], 0, j);
                float derivativeAhead = MAT_AT(g.layers[l], 0, j);
                float activationDerivative = 1.f;
                if (nn.activations)
                {
                    activationDerivative = getActDerivative(nn.activations[l].type)(outputAhead);
                }
                float fullDerivative = (s * derivativeAhead * activationDerivative);
                MAT_AT(g.biases[l - 1], 0, j) += (fullDerivative);

                for (size_t k = 0; k < nn.layers[l - 1].cols; k++)
                {
                    // j - weights matrix col
                    // k = weights matrix row
                    float prevInput = MAT_AT(nn.layers[l - 1], 0, k);
                    MAT_AT(g.weights[l - 1], k, j) += (fullDerivative * prevInput);

                    float prevWeight = MAT_AT(nn.weights[l - 1], k, j);
                    MAT_AT(g.layers[l - 1], 0, k) += (fullDerivative * prevWeight);
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; i++)
    {
        Matrix curWeights = g.weights[i];
        for (size_t j = 0; j < curWeights.rows; j++)
        {
            for (size_t k = 0; k < curWeights.cols; k++)
            {
                MAT_AT(curWeights, j, k) /= n;
            }
        }

        Matrix curBiases = g.biases[i];
        for (size_t k = 0; k < curBiases.cols; k++)
        {
            MAT_AT(curBiases, 0, k) /= n;
        }
    }
}

void Network_policy_gradient_backprop(Network nn, Network g, Step *steps[], size_t stepAmount)
{
    if (NETWORK_IN(nn).cols != steps[0]->state.cols)
        return;
    if (!Network_same(nn, g))
        return;
    size_t n = stepAmount; // amount of steps

    Network_clear(g);

    float cumulativeRewards[n];
    cumulativeRewards[n - 1] = steps[n - 1]->reward;
    for (int i = n - 2; i >= 0; i--)
    {
        cumulativeRewards[i] = steps[i]->reward + 0.9 * cumulativeRewards[i + 1];
    }

    for (size_t i = 0; i < n; i++)
    {
        mat_copy(NETWORK_IN(nn), steps[i]->state);
        Network_forward(nn);
        SOFTMAX_OUTPUTS(nn);

        for (size_t j = 0; j <= g.count; j++)
        {
            mat_clear(g.layers[j]);
        }

        for (size_t j = 0; j < NETWORK_OUT(nn).cols; j++)
        {
            float P_k = MAT_AT(NETWORK_OUT(nn), 0, j);
            if (steps[i]->action == j)
            {
                MAT_AT(NETWORK_OUT(g), 0, j) = (P_k - 1) * cumulativeRewards[i];
            }
            else
            {
                MAT_AT(NETWORK_OUT(g), 0, j) = P_k * cumulativeRewards[i];
            }
        }

        for (size_t l = nn.count; l > 0; l--)
        {
            for (size_t j = 0; j < nn.layers[l].cols; j++)
            {
                float outputAhead = MAT_AT(nn.layers[l], 0, j);
                float derivativeAhead = MAT_AT(g.layers[l], 0, j);
                float activationDerivative = 1.f;
                if (nn.activations)
                {
                    activationDerivative = getActDerivative(nn.activations[l - 1].type)(outputAhead);
                }
                float fullDerivative = (derivativeAhead * activationDerivative);
                MAT_AT(g.biases[l - 1], 0, j) += (fullDerivative);

                for (size_t k = 0; k < nn.layers[l - 1].cols; k++)
                {
                    // j - weights matrix col
                    // k = weights matrix row
                    float prevInput = MAT_AT(nn.layers[l - 1], 0, k);
                    MAT_AT(g.weights[l - 1], k, j) += (fullDerivative * prevInput);

                    float prevWeight = MAT_AT(nn.weights[l - 1], k, j);
                    MAT_AT(g.layers[l - 1], 0, k) += (fullDerivative * prevWeight);
                }
            }
        }
    }

    for (size_t i = 0; i < g.count; i++)
    {
        Matrix curWeights = g.weights[i];
        for (size_t j = 0; j < curWeights.rows; j++)
        {
            for (size_t k = 0; k < curWeights.cols; k++)
            {
                MAT_AT(curWeights, j, k) /= n;
            }
        }

        Matrix curBiases = g.biases[i];
        for (size_t k = 0; k < curBiases.cols; k++)
        {
            MAT_AT(curBiases, 0, k) /= n;
        }
    }
}

void Network_learn(Network nn, Network g, float rate)
{
    if (!Network_same(nn, g))
        return;

    for (size_t i = 0; i < nn.count; i++)
    {
        Matrix weights = nn.weights[i];
        for (size_t j = 0; j < weights.rows; j++)
        {
            for (size_t k = 0; k < weights.cols; k++)
            {
                MAT_AT(weights, j, k) -= rate * MAT_AT(g.weights[i], j, k);
            }
        }

        Matrix biases = nn.biases[i];
        for (size_t j = 0; j < biases.rows; j++)
        {
            for (size_t k = 0; k < biases.cols; k++)
            {
                MAT_AT(biases, j, k) -= rate * MAT_AT(g.biases[i], j, k);
            }
        }
    }
}

#endif // ML_H