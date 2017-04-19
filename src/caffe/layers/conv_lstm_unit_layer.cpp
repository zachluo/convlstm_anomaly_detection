#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/conv_lstm_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  return 2. * sigmoid(2. * x) - 1.;
}

/**
 * bottom[0]->shape(): 1 X N X D X H X W
 * bottom[1]->shape(): 1 X N X 4D X H X W
 * bottom[2]->shape(): 1 X N
 */
template <typename Dtype>
void ConvolutionLSTMUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int num_instances = bottom[0]->shape(1);
	//const int height = bottom[0]->shape(3);
	//const int width = bottom[0]->shape(4);
	for (int i = 0; i < bottom.size(); ++i) {
		if (i == 2) {
			CHECK_EQ(2, bottom[i]->num_axes());
		} else {
			CHECK_EQ(5, bottom[i]->num_axes());
			/*CHECK_EQ(height, bottom[i]->shape(3)); // 检查高度
			CHECK_EQ(width, bottom[i]->shape(4));  // 检查宽度*/
		}
		CHECK_EQ(1, bottom[i]->shape(0));
		CHECK_EQ(num_instances, bottom[i]->shape(1));
	}
	hidden_dim_ = bottom[0]->shape(2); 	// N_
	CHECK_EQ(4 * hidden_dim_, bottom[1]->shape(2)); // 4D
	top[0]->ReshapeLike(*bottom[0]); 		// top[0]->shape(): 1 x N x D x H x W
	top[1]->ReshapeLike(*bottom[0]); 		// top[1]->shape(): 1 x N x D x H x W
	X_acts_.ReshapeLike(*bottom[1]); 		// X_acts_.shape(): 1 x N x 4D x H x W
}

/**
 * bottom[0]->shape(): 1 X N X D X H X W
 * bottom[1]->shape(): 1 X N X 4D X H X W
 * bottom[2]->shape(): 1 X N
 *
 * top[0]->shape(): 1 x N x D x H x W
 * top[1]->shape(): 1 x N x D x H x W
 */
template <typename Dtype>
void ConvolutionLSTMUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const int num = bottom[0]->shape(1);
	const int x_dim = hidden_dim_ * 4;
	const int height = bottom[0]->shape(3);
	const int width = bottom[0]->shape(4);

	const Dtype* C_prev = bottom[0]->cpu_data(); // C_prev: 1 x N x D x H x W
	const Dtype* X = bottom[1]->cpu_data();		  // X: 1 x N x 4D x H x W
	const Dtype* cont = bottom[2]->cpu_data();   // cont: 1 x N

	Dtype* C = top[0]->mutable_cpu_data();			  // C: 1 x N x D x H x W
	Dtype* H = top[1]->mutable_cpu_data(); 				// H: 1 x N x D x H x W

	for (int n = 0; n < num; ++n) {
		for (int d = 0; d < hidden_dim_; ++d) {
			for (int h = 0; h < height; ++h) {
				for (int w = 0; w < width; ++w) {
					const int idx = h * width + w;
					const Dtype i = sigmoid(X[d + idx]);
					const Dtype f = (*cont == 0) ? 0 :
							(*cont * sigmoid(X[1 * hidden_dim_ + d + idx]));
					const Dtype o = sigmoid(X[2 * hidden_dim_ + d + idx]);
					const Dtype g = tanh(X[3 * hidden_dim_ + d + idx]);
					const Dtype c_prev = C_prev[d + idx];
					const Dtype c = f * c_prev + i * g;
					C[d + idx] = c;
					const Dtype tanh_c = tanh(c);
					H[d + idx] = o * tanh_c;
				}
			}
		}
		const int offset = height * width;
		C_prev += (hidden_dim_ + offset);
		X += (x_dim + offset);
		C += (hidden_dim_ + offset);
		H += (hidden_dim_ + offset);
		++cont;
	}
}

template <typename Dtype>
void ConvolutionLSTMUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

	const int num = bottom[0]->shape(1);
	const int x_dim = hidden_dim_ * 4;
	const int height = bottom[0]->shape(3);
	const int width = bottom[0]->shape(4);

  const Dtype* C_prev = bottom[0]->cpu_data();
  const Dtype* X = bottom[1]->cpu_data();
  const Dtype* cont = bottom[2]->cpu_data();
  const Dtype* C = top[0]->cpu_data();
  const Dtype* H = top[1]->cpu_data();
  const Dtype* C_diff = top[0]->cpu_diff();
  const Dtype* H_diff = top[1]->cpu_diff();

  Dtype* C_prev_diff = bottom[0]->mutable_cpu_diff();
  Dtype* X_diff = bottom[1]->mutable_cpu_diff();

  for (int n = 0; n < num; ++n) {
    for (int d = 0; d < hidden_dim_; ++d) {
    	for (int h = 0; h < height; ++h) {
    		for (int w = 0; w < width; ++w) {
    			const int idx = h * width + w;
          const Dtype i = sigmoid(X[d + idx]);
          const Dtype f = (*cont == 0) ? 0 :
              (*cont * sigmoid(X[1 * hidden_dim_ + d + idx]));
          const Dtype o = sigmoid(X[2 * hidden_dim_ + d + idx]);
          const Dtype g = tanh(X[3 * hidden_dim_ + d + idx]);
          const Dtype c_prev = C_prev[d + idx];
          const Dtype c = C[d + idx];
          const Dtype tanh_c = tanh(c);
          Dtype* c_prev_diff = C_prev_diff + d + idx;
          Dtype* i_diff = X_diff + d + idx;
          Dtype* f_diff = X_diff + 1 * hidden_dim_ + d + idx;
          Dtype* o_diff = X_diff + 2 * hidden_dim_ + d + idx;
          Dtype* g_diff = X_diff + 3 * hidden_dim_ + d + idx;
          const Dtype c_term_diff =
              C_diff[d + idx] + H_diff[d + idx] * o * (1 - tanh_c * tanh_c);
          *c_prev_diff = c_term_diff * f;
          *i_diff = c_term_diff * g * i * (1 - i);
          *f_diff = c_term_diff * c_prev * f * (1 - f);
          *o_diff = H_diff[d + idx] * tanh_c * o * (1 - o);
          *g_diff = c_term_diff * i * (1 - g * g);
    		}
    	}
    }
    const int offset = height * width;
    C_prev += (hidden_dim_ + offset);
    X += (x_dim + offset);
    C += (hidden_dim_ + offset);
    H += (hidden_dim_ + offset);
    C_diff += (hidden_dim_ + offset);
    H_diff += (hidden_dim_ + offset);
    X_diff += (x_dim + offset);
    C_prev_diff += (hidden_dim_ + offset);
    ++cont;
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLSTMUnitLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLSTMUnitLayer);
REGISTER_LAYER_CLASS(ConvolutionLSTMUnit);

}  // namespace caffe
