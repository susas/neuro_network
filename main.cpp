#include <stdlib.h>
#include <iostream>
#include <vector>
#include <chrono>

//#include <omp.h>

#include <assert.h>
#include <random>

#define DEBUG std::cerr << "Debug" << std::endl;

class matrix
{
public:

    matrix(size_t rows, size_t columns, const float& value = 0.f)
    {
        if(rows == 0 || columns == 0)
        {
            std::cerr << "[constructor] matrix(size_t rows, size_t columns): rows and columns must be non-zero" << std::endl;
            return;
        }
        this->vec.resize(rows);

        for(size_t i = 0; i < rows; i++)
            this->vec[i].resize(columns, value);
        
        this->_rows = rows;
        this->_columns = columns;
    }

    matrix(const std::vector<std::vector<float>>* mat)
    {
        this->_rows = mat->size();
        this->_columns = mat->at(0).size();
        // check if matrix is square
        for(size_t i = 0; i < this->rows; i++)
        {
            if(mat->at(i).size() != this->columns)
            {
                std::cerr << "[constructor] matrix(std::vector<std::vector<float>>* mat): Vector's rows must be equal." << std::endl;
                return;
            }
        }
        this->vec = *mat;
    }

    matrix(const matrix& obj)
    {
        this->_columns = obj.columns;
        this->_rows = obj.rows;
        this->vec = obj.vec;
    }

// Matrix multiplication

    matrix operator*(const matrix& obj)
    {
        if(this->columns != obj.rows)
        {
            std::cerr << "matrix operator*(const matrix& obj): Invalid dimensions for matrix multiplication. "
                << "this->columns= " << this->columns << " obj.rows= " << obj.columns << std::endl;
            return matrix(1,1);
        }

        matrix ret(this->rows, obj.columns);

        for(size_t i = 0; i < this->rows; i++)
            for(size_t j = 0; j < obj.columns; j++)
                for(size_t k = 0; k < this->columns; k++)
                    ret.vec[i][j] += this->vec[i][k] * obj.vec[k][j];

        return ret;
    }

// Addition

    matrix operator+(const matrix& obj)
    {
        if(this->rows != obj.rows)
        {
            std::cerr << "matrix operator+(const matrix& obj): Rows much match for addition. " << 
                std::endl << " this->rows: " << this->rows << std::endl << " obj.rows: " << obj.rows << std::endl;
            std::exit(-1);
            return matrix(1,1);
        }
            
        else if(this->columns != obj.columns)
        {
            std::cerr << "matrix operator+(const matrix& obj): Columns much match for addition. " << 
                std::endl << " this->columns: " << this->columns << std::endl << " obj.columns: " << obj.columns << std::endl;
            std::exit(-1);
            return matrix(1,1);
        }
            
        else
        {
            matrix ret(this->rows, this->columns);
            
            for(size_t i = 0; i < this->rows; i++)
                for(size_t j = 0; j < this->columns; j++)
                    ret.vec[i][j] = this->vec[i][j] + obj.vec[i][j]; 
            return ret;
        }
    }

    void operator+=(const matrix& obj)
    {
        if(this->rows != obj.rows)
            throw "Rows much match for addition";
        else if(this->columns != obj.columns)
            throw "Columns much match for addition";
        else
        {
            
            for(size_t i = 0; i < this->rows; i++)
                for(size_t j = 0; j < this->columns; j++)
                    this->vec[i][j] += obj.vec[i][j]; 
        }
    }

// Subtraction

    matrix operator-(const matrix& obj)
    {
        if(this->rows != obj.rows)
        {
            std::cerr << "matrix operator-(const matrix& obj): Rows much match for subtraction" << std::endl;
            return NULL;      
        }
        else if(this->columns != obj.columns)
        {
            std::cerr << "matrix operator-(const matrix& obj): Columns much match for subtraction" << std::endl;
            return NULL;  
        }
        else
        {
            matrix ret(this->rows, this->columns);
            
            for(size_t i = 0; i < this->rows; i++)
                for(size_t j = 0; j < this->columns; j++)
                    ret.vec[i][j] = this->vec[i][j] - obj.vec[i][j]; 
            return ret;
        }
    }

    matrix operator-=(const matrix& obj)
    {
        if(this->rows != obj.rows)
        {
            std::cerr << "matrix operator-=(const matrix& obj): Rows much match for subtraction" << std::endl;
            return NULL;       
        }
        else if(this->columns != obj.columns)
        {
            std::cerr << "matrix operator-=(const matrix& obj): Columns much match for subtraction" << std::endl;
            return NULL;    
        }
        else
        {
            
            for(size_t i = 0; i < this->rows; i++)
                for(size_t j = 0; j < this->columns; j++)
                    this->vec[i][j] -= obj.vec[i][j]; 
                    
            
        }
    }

    void operator=(const matrix& obj)
    {
        if(this->rows != obj.rows)
        {
            std::cerr << "void operator=(const matrix& obj): Rows don't match" << std::endl;
            return;       
        }
        else if(this->columns != obj.columns)
        {
            std::cerr << "void operator=(const matrix& obj): Columns don't match" << std::endl;
            return;    
        }
        else
        {
            this->vec = obj.vec;
        }
            
    }

// Other functionality
  
    void vercat(const matrix obj)
    {
        if(this->columns != obj.columns)
        {
            std::cerr << "void vercat(const matrix obj): columns do not match" << std::endl;
            return;
        }
        else
        {
            this->vec.resize(this->rows + obj.rows, std::vector<float>(0.f));
            for(size_t i = this->rows; i < this->rows + obj.rows; i++)
                this->vec[i] = obj.vec[i-this->rows];
            this->_rows += obj.rows;
        }
    }

    void horcat(const matrix& obj)
    {
        if(this->rows != obj.rows)
        {
            std::cerr << "void horcat(const matrix obj): rows do not match" << std::endl;
            return;
        }
        else
        {
            for(size_t i = 0; i < this->rows; i++)
            {
                this->vec[i].resize(this->columns + obj.columns, 0.f);
                for(size_t j = this->columns; j < this->columns + obj.columns; j++)
                    this->vec[i][j] = obj.vec[i][j-this->columns];
            } 
            this->_columns += obj.columns;
        }
    }

    void remove_row(size_t start, size_t end = 0)
    {
        if(start+end > this->rows-1)
        {
            std::cerr << "void remove_row(size_t row): Row is out of bounds. start=" << start << " end=" << end << std::endl;
            return;
        }
        else if(start > end && end != 0)
        {
            std::cerr << "void remove_row(size_t row): end must be bigger than start. start=" << start << " end=" << end << std::endl;
            return;
        }
        else
        {
            if(end == 0)
            {
                this->vec.erase(this->vec.begin()+start);
                this->_rows--;
            } 
            else
            {
                this->vec.erase(this->vec.begin() + start, this->vec.begin() + end);
                this->_rows -= start+end;
            }    
        }
    }

    void remove_column(size_t start, size_t end = 0)
    {
        if(start+end > this->columns-1)
        {
            std::cerr << "void remove_column(size_t column): Column is out of bounds. start=" << start << " end=" << end << std::endl;
            return;
        }
        else if(start > end && end != 0)
        {
            std::cerr << "void remove_column(size_t column): end must be bigger than start. column=" << start << std::endl;
            return;
        }
        else
        {   
            if(end == 0)
                for(auto& row : this->vec)
                    row.erase(row.begin() + start);
            else
                for(auto& row : this->vec)
                    row.erase(row.begin() + start, row.begin() + end);
            this->_columns -= start+end;
        }
    }

    void transpose()
    {
        std::vector<std::vector<float>> hold = this->vec;
        this->vec.resize(this->columns, std::vector<float>(this->columns, 1));
        for(size_t i = 0; i < this->columns; i++)
        {
            this->vec[i].resize(this->rows);
            for(size_t j = 0; j < this->rows; j++)   
                this->vec[i][j] = hold[j][i];
        }
        // Swap rows and columns
        this->_rows += this->_columns;
        this->_columns = this->_rows - this->_columns;
        this->_rows -= this->_columns;
    }

    void random(const float& min = -1.f, const float& max = 1.f)
    {
        
        for(auto &row : this->vec)
            for(auto &column : row)
                column = (float)rand() / (float)RAND_MAX * (max-min)+min;
    }

    void identity()
    {
        if(this->rows != this->columns)
        {
            std::cerr << "void identity(): Matrix is not a square matrix" << std::endl;
            return;
        }
        for(size_t i = 0; i < this->rows; i++)
            for(size_t j = 0; j < this->columns; j++)
                (i == j) ?
                this->vec[i][j] = 1.0f :
                this->vec[i][j] = 0.0f;
    }

    void print()
    {
        for(auto& row : this->vec)
        {
            for(auto &column : row)
                std::cout << column << " ";
            std::cout << std::endl;
        }   
    }

    void sigmoid()
    {
        //
        for(auto &row : this->vec)
            for(auto &column : row)
                column = 1/(1+expf(-column));
    }

    matrix get_row(size_t row)
    {
        if(row >= this->rows)
        {
            std::cerr << "matrix get_row(size_t row): Row is out of bounds. row=" << row << std::endl;
            return matrix(1,1);
        }
        else
        {
            std::vector<std::vector<float>> t = {this->vec[row]};
            return matrix(&t);
        }
    }

    matrix get_column(size_t column)
    {
        if(column >= this->columns)
        {
            std::cerr << "matrix get_column(size_t column): column is out of bounds. column=" << column << std::endl;
            return matrix(1, 1);
        }
        else
        {
            matrix ret(this->rows,1);
            for(size_t i = 0; i < this->rows; i++)
                ret.vec[i][0] = this->vec[i][column];
            return ret;
        }
    }

    float get_value(size_t row, size_t column)
    {
        if(row > this->rows)
        {
            std::cerr << "float get_value(size_t row, size_t column): Row is out of bounds" << " row=" << row << std::endl;
            return -1.1111111f;
        }
        else if(column > this->columns)
        {
            std::cerr << "float get_value(size_t row, size_t column): Column is out of bounds" << " column=" << column << std::endl;
            return -1.1111111f;
        }
        else
            return this->vec[row][column];
    }

    void set_value(size_t row, size_t column, const float& value)
    {
        if(row > this->rows)
        {
            std::cerr << "float set_value(size_t row, size_t column): Row is out of bounds" << " row=" << row << std::endl;
            return;
        }
        else if(column > this->columns)
        {
            std::cerr << "float set_value(size_t row, size_t column): Column is out of bounds" << " column=" << column << std::endl;
            return;
        }
        else
            this->vec[row][column] = value;
    }

    const size_t& rows = this->_rows;
    const size_t& columns = this->_columns;

    
private:

    std::vector<std::vector<float>> vec;
    size_t _rows, _columns; 
        
};

class neuronetwork
{
public:

    neuronetwork(std::vector<size_t> arch)
    {
        this->_length = arch.size();
        this->_arch = arch;
        
        for(size_t i = 0; i < arch.size()-1; i++)
        {
            this->_nodes.push_back(matrix(1, arch[i]));
            this->_biases.push_back(matrix(1, arch[i+1]));
            this->_weights.push_back(matrix(arch[i], arch[i+1]));
        }
        this->_nodes.push_back(matrix(1, arch[arch.size()-1]));
    }

    void print_arch()
    {
        std::cout << "Num of nodes: " << this->_nodes.size() << std::endl
             << "Num of weights: " << this->_weights.size() << std::endl
             << "Num of biases: " << this->_biases.size() << std::endl;
        std::cout << "Weights: ";
        for(auto &weight : _weights)
        {
            std::cout << "(" << weight.rows << "," << weight.columns << ")" << " -> ";
        }
        std::cout << std::endl;

        std::cout << "Biases: ";
        for(auto &bias : _biases)
        {
            std::cout << "(" << bias.rows << "," << bias.columns << ")" << " -> ";
        }
        std::cout << std::endl;

        std::cout << "Nodes: ";
        for(auto &node : _nodes)
        {
            std::cout << "(" << node.rows << "," << node.columns << ")" << " -> ";
        }
        std::cout << std::endl;
        
    }

    void randomize_model(float min = -1.f, float max = 1.f)
    {
        for(auto &weight : _weights)
            weight.random(min, max);
        for(auto &bias : _biases)
            bias.random(min, max);
    }

    void forward()
    {
        this->outputs.clear();
        for(size_t i = 0; i < this->inputs->rows; i++)
        {
            this->_nodes[0] = this->inputs->get_row(i);
            for(size_t i = 0; i < this->_length-1; i++)
            {
                this->_nodes[i+1] = (this->_nodes[i]*this->_weights[i]) + this->_biases[i];
                this->_nodes[i+1].sigmoid();
            }
            this->outputs.push_back(this->get_output());
        }
    }

    float loss(matrix expected)
    {
        float error = 0.f;
        //#pragma omp parallel for
        for(size_t i = 0; i < outputs.size(); i++)
        {
            error += (this->outputs[i] - expected.get_value(i, 0))*(this->outputs[i] - expected.get_value(i, 0));
        }
        error /= this->outputs.size();
        return error;
    }

    void train(matrix expected, size_t epochs = 3, float eps=1e-4)
    {
        std::vector<matrix> Weights = this->_weights;
        std::vector<matrix> Biases = this->_biases;
        
        for(size_t epochs_ = 0; epochs_ < epochs; epochs_++)
        {
            forward();
            float error = this->loss(expected);
            
            
            for(size_t k = 0; k < this->_weights.size(); k++)
            {
                for(size_t i = 0; i < this->_weights[k].rows; i++)
                {
                    for(size_t j = 0; j < this->_weights[k].columns; j++)
                    {
                        this->_weights[k].set_value(i, j, this->_weights[k].get_value(i, j) + eps);
                        //this->_weights[k].vec[i][j] += eps;
                        forward();
                        //Weights[k].vec[i][j] -= (this->loss(expected) - error)/eps;
                        Weights[k].set_value(i, j, Weights[k].get_value(i, j) - (this->loss(expected) - error)/eps);
                        //this->_weights[k].vec[i][j] -= eps;
                        this->_weights[k].set_value(i, j, this->_weights[k].get_value(i, j) - eps);
                    }
                }
            }
            
            for(size_t k = 0; k < this->_biases.size(); k++)
            {
                for(size_t i = 0; i < this->_biases[k].rows; i++)
                {
                    for(size_t j = 0; j < this->_biases[k].columns; j++)
                    {
                        this->_biases[k].set_value(i, j, this->_biases[k].get_value(i, j) + eps);
                        //this->_biases[k].vec[i][j] += eps;
                        forward();
                        //Biases[k].vec[i][j] -= (this->loss(expected) - error)/eps;
                        Biases[k].set_value(i, j, Biases[k].get_value(i, j) - (this->loss(expected) - error)/eps);
                        //this->_biases[k].vec[i][j] -= eps;
                        this->_biases[k].set_value(i, j, this->_biases[k].get_value(i, j) - eps);
                    }
                }
            }
            
            this->_weights = Weights;
            this->_biases = Biases;
        }
    }

    matrix* inputs;
    std::vector<float> outputs;

private:

    float get_output()
    {
        return this->_nodes[this->_nodes.size()-1].get_value(0,0);
    }

    std::vector<size_t> _arch;

    std::vector<matrix> _weights;
    std::vector<matrix> _biases;
    std::vector<matrix> _nodes;

    size_t _length; 
};


int main(void)
{
    srand(time(0));

    /* 
        XOR gate's inputs in the first two columns 
        and the outputs of XOR gate in the last column.
    */

    std::vector<std::vector<float>> f_data = {
        {0.f, 0.f, 0.f},
        {0.f, 1.f, 1.f},
        {1.f, 0.f, 1.f},
        {1.f, 1.f, 0.f}
    };
    matrix data(&f_data);

    /* Processing the data into x_train and y_train. */
    matrix x_train = data;
    x_train.remove_column(x_train.columns-1);

    matrix y_train = data.get_column(data.columns-1);


    /* Creating and initializing the neuralnetwork. */
    neuronetwork n({2, 2, 1});
    n.randomize_model();
    n.inputs = &x_train;

    /* Timing the training. */
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < 1e4; i++)
    {
        
        n.train(y_train, 1, 1e-3);
        
        /*Logs loss*/
        std::cout << "loss: " << n.loss(y_train) << std::endl; 

    }
    /* Printing the time it took to train. */
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (stop-start);
    std::cout << "It took: " << duration.count()/10 << "ms" << std::endl;

    /* Outputting the models accuracy. */
    for(auto &output : n.outputs)
        std::cout << output << std::endl;


    return 0;
}
