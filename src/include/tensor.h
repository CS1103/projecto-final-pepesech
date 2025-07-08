//
// Created by valma on 5/11/2025.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <iterator>
#include <concepts>
#include <iostream>
#include <type_traits>
#include <cassert>



namespace utec::algebra {

    template<typename T>
    concept arithmetic = std::integral<T> or std::floating_point<T>;

    template<typename T, size_t Rank>
    class Tensor {
        std::array<size_t, Rank> shape_;
        std::array<size_t, Rank> strides_;
        std::vector<T> data_;

    public:
        Tensor() = default;
        static std::array<size_t, Rank> calc_strides(const std::array<size_t, Rank> &shape) {
            std::array<size_t, Rank> strides{};
            size_t stride = 1;
            for (int i = Rank - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        static constexpr size_t total_size(const std::array<size_t, Rank> &shape) {
            return std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});
        }

        explicit Tensor(const std::array<size_t, Rank> &shape) : shape_(shape) {
            strides_=calc_strides(shape_);
            data_ = std::vector<T>(total_size(shape_));
        }

        Tensor(const std::initializer_list<size_t> &shape) {
            if (shape.size() != Rank) {
                throw std::invalid_argument("Initializer list size must match Rank");
            }
            std::copy(shape.begin(), shape.end(), shape_.begin());
            strides_ = calc_strides(shape_);
            data_ = std::vector<T>(total_size(shape_));
        }

        Tensor(const std::vector<std::vector<T>>& data_vec) {
            if (data_vec.empty() || data_vec[0].empty())
                throw std::invalid_argument("Data vector is empty or malformed.");

            shape_ = {data_vec.size(), data_vec[0].size()};
            data_.resize(shape_[0] * shape_[1]);

            for (size_t i = 0; i < shape_[0]; ++i)
                for (size_t j = 0; j < shape_[1]; ++j)
                    (*this)(i, j) = data_vec[i][j];
        }

        template<typename... Dims> requires (std::conjunction_v<std::is_convertible<Dims, size_t>...>)
        explicit Tensor(Dims... dims) {
            if constexpr (sizeof...(Dims) != Rank) {
                throw std::invalid_argument("Number of dimensions do not match with "+std::to_string(Rank));
            }else {
                shape_ = {static_cast<size_t>(dims)...};
                strides_ = calc_strides(shape_);
                data_ = std::vector<T>(total_size(shape_));
            }
        }

        Tensor(const std::array<size_t, Rank>& shape, T value) : shape_(shape) {
            strides_ = calc_strides(shape_);
            data_ = std::vector<T>(total_size(shape_), value);
        }

        size_t size() const {
            return data_.size();
        }

        template<typename... Dims>
        void reshape(Dims... dims) {
            std::vector<size_t> new_dims = {static_cast<size_t>(dims)...};

            if (new_dims.size() != Rank) {
                throw std::invalid_argument("Number of dimensions do not match with " + std::to_string(Rank));
            }

            std::array<size_t, Rank> shape;
            std::copy(new_dims.begin(), new_dims.end(), shape.begin());

            size_t total = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>{});

            shape_=shape;
            strides_=calc_strides(shape_);
            data_.resize(total);
        }

        std::array<size_t, Rank> shape() const { return shape_; }

        size_t shape(size_t i) const {return shape_[i];}

        void fill(const T &n) {
            std::fill(data_.begin(), data_.end(), n);
        }

        Tensor &operator =(std::initializer_list<T> ilist) {
            if (ilist.size() != data_.size()) {
                throw std::invalid_argument("Data size does not match tensor size");
            }
            std::copy(ilist.begin(), ilist.end(), data_.begin());
            return *this;
        }

        size_t compute_index(const std::array<size_t, Rank>& indexes) const {
            size_t linear_index = 0;
            for(size_t i = 0; i < Rank; i++) {
                if (indexes[i] >= shape_[i]) {
                    throw std::out_of_range("Out of range");
                }
                linear_index += indexes[i]* strides_[i];
            }
            return linear_index;
        }

        template<typename... Idxs>
        T &operator()(Idxs... idxs) {
            std::array<size_t, Rank> indexes{static_cast<size_t>(idxs)...};
            size_t linear_idx = compute_index(indexes);
            return data_[linear_idx];
        }

        template<typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            std::array<size_t, Rank> indices{static_cast<size_t>(idxs)...};
            size_t linear_index = compute_index(indices);
            return data_[linear_index];
        }


        Tensor& operator =(Tensor other) {
            data_ = other.data_;
            shape_ = other.shape_;
            strides_ = other.strides_;

            return *this;
        }

        template<typename Func>
        Tensor apply_binary_op(const Tensor& other, Func op) const {
            if constexpr (Rank != Rank) {
                throw std::invalid_argument("Ranks do not match");
            }

            std::array<size_t, Rank> result_shape;
            for (size_t i = 0; i < Rank; ++i) {
                if (shape_[i] == other.shape_[i]) {
                    result_shape[i] = shape_[i];
                } else if (shape_[i] == 1) {
                    result_shape[i] = other.shape_[i];
                } else if (other.shape_[i] == 1) {
                    result_shape[i] = shape_[i];
                } else {
                    throw std::invalid_argument("Shapes do not match and they are not compatible for broadcasting");
                }
            }

            Tensor result(result_shape);

            std::array<size_t, Rank> idx{};
            size_t total = Tensor::total_size(result_shape);

            for (size_t linear = 0; linear < total; ++linear) {
                size_t rem = linear;
                for (int i = Rank - 1; i >= 0; --i) {
                    idx[i] = rem % result_shape[i];
                    rem /= result_shape[i];
                }

                std::array<size_t, Rank> idx_a, idx_b;
                for (size_t i = 0; i < Rank; ++i) {
                    idx_a[i] = (shape_[i] == 1) ? 0 : idx[i];
                    idx_b[i] = (other.shape_[i] == 1) ? 0 : idx[i];
                }

                result(idx) = op((*this)(idx_a), other(idx_b));
            }

            return result;
        }

        Tensor<T,2> slice(size_t row_start, size_t row_end) const {
            assert(row_end <= shape_[0] && row_start < row_end);

            Tensor<T,2> result({row_end - row_start, shape_[1]});
            for (size_t i = row_start; i < row_end; ++i)
                for (size_t j = 0; j < shape_[1]; ++j)
                    result(i - row_start, j) = (*this)(i, j);
            return result;
        }

        Tensor operator+(const Tensor& other) {
            return apply_binary_op(other, [](T a, T b){return a + b;});
        }

        Tensor operator-(const Tensor& other) {
            return apply_binary_op(other, [](T a, T b){return a - b;});
        }

        Tensor operator*(const Tensor& other) {
            return apply_binary_op(other, [](T a, T b){return a * b;});
        }

        Tensor operator/(const Tensor& other) {
            return apply_binary_op(other, [](T a, T b){return a / b;});
        }


        const T& operator()(const std::array<size_t, Rank>& indices) const {
            size_t linear_index = compute_index(indices);
            return data_[linear_index];
        }

        T& operator()(const std::array<size_t, Rank>& indices){
            size_t linear_index = compute_index(indices);
            return data_[linear_index];
        }

        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }

        auto cbegin() const { return data_.cbegin(); }
        auto cend() const { return data_.cend(); }


        template<typename num> requires arithmetic<num>
        Tensor operator +(num k) {
            Tensor result(shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]= data_[i] +k;
            }
            return result;
        }

        template<typename num> requires arithmetic<num>
        Tensor operator -(num k) {
            Tensor result(shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]= data_[i]-k;
            }
            return result;
        }


        template<typename num> requires arithmetic<num>
        Tensor operator *(num k) {
            Tensor result(shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i] = data_[i]*k;
            }
            return result;
        }


        template<typename num> requires arithmetic<num>
        Tensor operator /(num k) {
            Tensor result(shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i] = data_[i]/k;
            }
            return result;
        }


        template<typename num> requires arithmetic<num>
        friend Tensor operator+(num k, Tensor tensor) {
            Tensor result(tensor.shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]=tensor.data_[i]+k;
            }
            return result;
        }

        template<typename num> requires arithmetic<num>
        friend Tensor operator-(num k, Tensor tensor) {
            Tensor result(tensor.shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]=k-tensor.data_[i];
            }
            return result;
        }

        template<typename num> requires arithmetic<num>
        friend Tensor operator*(num k, Tensor tensor) {
            Tensor result(tensor.shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]=tensor.data_[i]*k;
            }
            return result;
        }

        template<typename num> requires arithmetic<num>
        friend Tensor operator/(num k, Tensor tensor) {
            Tensor result(tensor.shape_);
            for(size_t i = 0; i<result.data_.size(); i++) {
                result.data_[i]=k/tensor.data_[i];
            }
            return result;
        }


        template<size_t I>
        constexpr auto get() const -> std::enable_if_t<(I < Rank), size_t> {
            return shape_[I];
        }

        template<size_t I>
        constexpr auto get() -> std::enable_if_t<(I < Rank), size_t&> {
            return shape_[I];
        }
    };

    template <typename T, size_t Rank>
    void print_recursive(std::ostream& os, const Tensor<T, Rank>& tensor, std::array<size_t, Rank>& indices, size_t depth) {
        size_t dim = tensor.shape()[depth];

        if (depth == Rank - 1) {
            for (size_t i = 0; i < dim; ++i) {
                indices[depth] = i;
                os << tensor(indices);
                if (i + 1 < dim) os << " ";
            }
            os << "\n";
        } else {
            os << "{\n";
            for (size_t i = 0; i < dim; ++i) {
                indices[depth] = i;
                print_recursive(os, tensor, indices, depth + 1);
            }
            os << "}\n";
        }
    }

    template <typename T, size_t Rank>
    std::ostream& operator<<(std::ostream& os, const Tensor<T, Rank>& tensor) {

        std::array<size_t, Rank> indices{};
        print_recursive(os, tensor, indices, 0);

        return os;
    }

    template<typename T, size_t Rank>
Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& input) {
        if constexpr (Rank < 2) {
            throw std::invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
        }

        auto original_shape = input.shape();
        auto new_shape = original_shape;
        std::swap(new_shape[Rank - 2], new_shape[Rank - 1]);

        Tensor<T, Rank> result(new_shape);

        std::array<size_t, Rank> idx;
        idx.fill(0);
        size_t total = Tensor<T, Rank>::total_size(original_shape);

        for (size_t i = 0; i < total; ++i) {

            auto transposed_idx = idx;
            std::swap(transposed_idx[Rank - 2], transposed_idx[Rank - 1]);

            result(transposed_idx) = input(idx);

            for (int j = Rank - 1; j >= 0; --j) {
                if (++idx[j] < original_shape[j]) break;
                idx[j] = 0;
            }
        }

        return result;
    }

    template<typename T, size_t Rank>
    Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& A, const Tensor<T, Rank>& B) {

        const auto& shape_a = A.shape();
        const auto& shape_b = B.shape();

        size_t M = shape_a[Rank - 2];
        size_t K_a = shape_a[Rank - 1];
        size_t K_b = shape_b[Rank - 2];
        size_t N = shape_b[Rank - 1];

        if (K_a != K_b) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication");
        }

        for (size_t i = 0; i < Rank - 2; ++i) {
            if (shape_a[i] != shape_b[i]) {
                throw std::invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
            }
        }

        std::array<size_t, Rank> result_shape = shape_a;
        result_shape[Rank - 2] = M;
        result_shape[Rank - 1] = N;

        Tensor<T, Rank> result(result_shape);
        std::array<size_t, Rank> idx{};
        idx.fill(0);

        size_t total_batches = 1;
        for (size_t i = 0; i < Rank - 2; ++i) {
            total_batches *= shape_a[i];
        }

        for (size_t b = 0; b < total_batches; ++b) {
            std::array<size_t, Rank - 2> batch_idx{};
           size_t rem = b;
            if constexpr(Rank >=3) {
                for (int i = Rank - 3; i >= 0; --i) {
                    batch_idx[i] = rem % shape_a[i];
                    rem /= shape_a[i];
                }
            }

           for (size_t i = 0; i < M; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    T sum = T{};
                    for (size_t k = 0; k < K_a; ++k) {
                        std::array<size_t, Rank> idx_a;
                        std::array<size_t, Rank> idx_b;

                        for (size_t d = 0; d < Rank - 2; ++d) {
                            idx_a[d] = idx_b[d] = batch_idx[d];
                        }
                        idx_a[Rank - 2] = i;
                        idx_a[Rank - 1] = k;

                        idx_b[Rank - 2] = k;
                        idx_b[Rank - 1] = j;

                        sum += A(idx_a) * B(idx_b);
                    }

                    std::array<size_t, Rank> idx_r;
                    for (size_t d = 0; d < Rank - 2; ++d)
                        idx_r[d] = batch_idx[d];
                    idx_r[Rank - 2] = i;
                    idx_r[Rank - 1] = j;

                    result(idx_r) = sum;
                }
            }
        }

    return result;
}

    template<typename T, size_t Rank, typename Func>
    Tensor<T, Rank> apply(const Tensor<T, Rank>& tensor, Func func) {
        Tensor<T, Rank> result(tensor.shape());

        auto it_src = tensor.cbegin();
        auto it_dst = result.begin();

        for (; it_src != tensor.cend(); ++it_src, ++it_dst) {
            *it_dst = func(*it_src);
        }

        return result;
    }

}

namespace std {
    template<typename T, size_t Rank>
    struct tuple_size<utec::algebra::Tensor<T, Rank>> : std::integral_constant<size_t, Rank> {};

    template<size_t I, typename T, size_t Rank>
    struct tuple_element<I, utec::algebra::Tensor<T, Rank>> {
        using type = size_t;
    };
}

#endif //TENSOR_H