#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/conv_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLSTMLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_0";
  (*names)[1] = "c_0";
}

template <typename Dtype>
void ConvolutionLSTMLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "h_" + format_int(this->T_);
  (*names)[1] = "c_T";
}

/**
 * format shapes into:
 * 	(h0) = shapes[0]: 1 x N x D x H x W
 * 	(c0) = shapes[1]: 1 x N x D x H x w
 */
template <typename Dtype>
void ConvolutionLSTMLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_blobs = 2; // h0, c0
	const int num_output = this->layer_param_.recurrent_param().num_output();

	const ConvolutionParameter& wxc_conv_param =
			this->layer_param_.recurrent_param().wxc_convolution_param();
//	const ConvolutionParameter& whh_conv_param =
//			this->layer_param_.recurrent_param().whh_convolution_param();

	int H_out_ = (this->H_ - wxc_conv_param.kernel_size(0) + 2 * wxc_conv_param.pad(0)) / wxc_conv_param.stride(0) + 1;
	int W_out_ = (this->W_ - wxc_conv_param.kernel_size(0) + 2 * wxc_conv_param.pad(0)) / wxc_conv_param.stride(0) + 1;

  shapes->resize(num_blobs);
  for (int i = 0; i < num_blobs; ++i) {
    (*shapes)[i].Clear();
    (*shapes)[i].add_dim(1); 	// a single timestep
    (*shapes)[i].add_dim(this->N_);  	// N
    (*shapes)[i].add_dim(num_output); // D
    // updates 2016-10-04
    (*shapes)[i].add_dim(H_out_);  	// H_out_
    (*shapes)[i].add_dim(W_out_);		// W_out_
  }
}

template <typename Dtype>
void ConvolutionLSTMLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h";
}

template <typename Dtype>
void ConvolutionLSTMLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {

	//LOG(INFO) << "haha 1";

	const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";

	const ConvolutionParameter& wxc_conv_param =
			this->layer_param_.recurrent_param().wxc_convolution_param();
	const ConvolutionParameter& whh_conv_param =
			this->layer_param_.recurrent_param().whh_convolution_param();

  /*CHECK_EQ(this->H_, (this->H_ - wxc_conv_param.kernel_size(0) + 2 * wxc_conv_param.pad(0)) /
  		wxc_conv_param.stride(0) + 1) << "wxc_convolution_param must keep height unchanged.";
  CHECK_EQ(this->W_, (this->W_ - wxc_conv_param.kernel_size(0) + 2 * wxc_conv_param.pad(0)) /
  		wxc_conv_param.stride(0) + 1) << "wxc_convolution_param must keep width unchanged.";
  CHECK_EQ(this->H_, (this->H_ - whh_conv_param.kernel_size(0) + 2 * whh_conv_param.pad(0)) /
  		whh_conv_param.stride(0) + 1) << "whh_convolution_param must keep height unchanged.";
  CHECK_EQ(this->W_, (this->W_ - whh_conv_param.kernel_size(0) + 2 * whh_conv_param.pad(0)) /
  		whh_conv_param.stride(0) + 1) << "whh_convolution_param must keep width unchanged.";*/

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter wxc_biased_conv_param;
  wxc_biased_conv_param.set_type("Convolution");
  wxc_biased_conv_param.mutable_convolution_param()->CopyFrom(wxc_conv_param);
  wxc_biased_conv_param.mutable_convolution_param()->set_num_output(num_output * 4);
  wxc_biased_conv_param.mutable_convolution_param()->set_bias_term(true);
  wxc_biased_conv_param.mutable_convolution_param()->set_axis(2);

  LayerParameter whh_unbiased_conv_param;
  whh_unbiased_conv_param.set_type("Convolution");
  whh_unbiased_conv_param.mutable_convolution_param()->CopyFrom(whh_conv_param);
  whh_unbiased_conv_param.mutable_convolution_param()->set_num_output(num_output * 4);
  whh_unbiased_conv_param.mutable_convolution_param()->set_bias_term(false);
  whh_unbiased_conv_param.mutable_convolution_param()->set_axis(2);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  LayerParameter split_param;
  split_param.set_type("Split");

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(2, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();

  input_layer_param->add_top("c_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[1]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xc_x = W_xc * x + b_c
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(wxc_biased_conv_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xc");
    x_transform_param->add_param()->set_name("b_c");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xc_x");
    x_transform_param->add_propagate_down(true);
  }

  if (this->static_input_) {
  	// unimplemented
  }

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->add_bottom("W_xc_x");
  x_slice_param->set_name("W_xc_x_slice");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("h_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("h");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xc_x_" + ts);

    // Add layers to flush the hidden state when beginning a new
    // sequence, as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    // Add layer to compute
    //     W_hc_h_{t-1} := W_hc * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(whh_unbiased_conv_param);
      w_param->set_name("transform_" + ts);
      w_param->add_param()->set_name("W_hc");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hc_h_" + tm1s);
    }

    // Add the outputs of the linear transformations to compute the gate input.
    //     gate_input_t := W_hc * h_conted_{t-1} + W_xc * x_t + b_c
    //                   = W_hc_h_{t-1} + W_xc_x_t + b_c
    {
      LayerParameter* input_sum_layer = net_param->add_layer();
      input_sum_layer->CopyFrom(sum_param);
      input_sum_layer->set_name("gate_input_" + ts);
      input_sum_layer->add_bottom("W_hc_h_" + tm1s);
      input_sum_layer->add_bottom("W_xc_x_" + ts);
      if (this->static_input_) {
        input_sum_layer->add_bottom("W_xc_x_static");
      }
      input_sum_layer->add_top("gate_input_" + ts);
    }

    // Add LSTMUnit layer to compute the cell & hidden vectors c_t and h_t.
    // Inputs: c_{t-1}, gate_input_t = (i_t, f_t, o_t, g_t), cont_t
    // Outputs: c_t, h_t
    //     [ i_t' ]
    //     [ f_t' ] := gate_input_t
    //     [ o_t' ]
    //     [ g_t' ]
    //         i_t := \sigmoid[i_t']
    //         f_t := \sigmoid[f_t']
    //         o_t := \sigmoid[o_t']
    //         g_t := \tanh[g_t']
    //         c_t := cont_t * (f_t .* c_{t-1}) + (i_t .* g_t)
    //         h_t := o_t .* \tanh[c_t]
    {
      LayerParameter* conv_lstm_unit_param = net_param->add_layer();
      conv_lstm_unit_param->set_type("ConvolutionLSTMUnit");
      conv_lstm_unit_param->add_bottom("c_" + tm1s);
      conv_lstm_unit_param->add_bottom("gate_input_" + ts);
      conv_lstm_unit_param->add_bottom("cont_" + ts);
      conv_lstm_unit_param->add_top("c_" + ts);
      conv_lstm_unit_param->add_top("h_" + ts);
      conv_lstm_unit_param->set_name("unit_" + ts);
    }
    output_concat_layer.add_bottom("h_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  {
    LayerParameter* c_T_copy_param = net_param->add_layer();
    c_T_copy_param->CopyFrom(split_param);
    c_T_copy_param->add_bottom("c_" + format_int(this->T_));
    c_T_copy_param->add_top("c_T");
  }
  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(ConvolutionLSTMLayer);
REGISTER_LAYER_CLASS(ConvolutionLSTM);

}  // namespace caffe
