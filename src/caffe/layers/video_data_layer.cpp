#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/video_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.video_data_param().new_height();
  const int new_width  = this->layer_param_.video_data_param().new_width();
  const bool is_color  = this->layer_param_.video_data_param().is_color();
  string root_folder = this->layer_param_.video_data_param().root_folder();
  const int T = this->layer_param_.video_data_param().t();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.video_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  //size_t pos;
  //int label;
  while (std::getline(infile, line)) {
    lines_.push_back(line);
  }

  CHECK(!lines_.empty()) << "File is empty";

  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    ShuffleVideos();
  }
  LOG(INFO) << "A total of " << lines_.size() << " videos.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read a video, and use it to initialize the top blob.
  cv::VideoCapture capture(root_folder + lines_[lines_id_]);
  CHECK(capture.isOpened()) << "Could not open the video " << lines_[lines_id_];
  cv::Mat cv_img;
  CHECK(capture.read(cv_img)) << "Could not read the video " << lines_[lines_id_];
  if(!is_color)
	  cv::cvtColor(cv_img, cv_img, CV_BGR2GRAY);
  cv::resize(cv_img,cv_img,cv::Size(new_width, new_height));
  //CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape.insert(top_shape.begin(),1);
  top_shape[0] = T;
  top_shape[1] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
//  top_shape[0] -= 1;
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->LegacyShape(0) << ","
      << top[0]->LegacyShape(1) << "," << top[0]->LegacyShape(2) << ","
      << top[0]->LegacyShape(3) << "," << top[0]->LegacyShape(4);
  // cont
  vector<int> label_shape(2);
  label_shape[0] = T;
  label_shape[1] = batch_size;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
  //label_shape[0] -= 1;
  //top[1]->Reshape(label_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void VideoDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const bool is_color = video_data_param.is_color();
  string root_folder = video_data_param.root_folder();
  const int T = this->layer_param_.video_data_param().t();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::VideoCapture capture(root_folder + lines_[lines_id_]);
  CHECK(capture.isOpened()) << "Could not open the video " << lines_[lines_id_];
  cv::Mat cv_img;
  CHECK(capture.read(cv_img)) << "Could not read the video " << lines_[lines_id_];
  if(!is_color)
	  cv::cvtColor(cv_img, cv_img, CV_BGR2GRAY);
  cv::resize(cv_img,cv_img,cv::Size(new_width, new_height));
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape.insert(top_shape.begin(),1);
  top_shape[0] = T;
  top_shape[1] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  //rng
  caffe::rng_t* prefetch_rng = static_cast<caffe::rng_t*>(prefetch_rng_->generator());

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
	CHECK_GT(lines_size, lines_id_);
	cv::VideoCapture capture(root_folder + lines_[lines_id_]);
	CHECK(capture.isOpened()) << "Could not open the video " << lines_[lines_id_];
	int num_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
	boost::uniform_int<> ui1(1, 3);
	int interval;
	if(this->layer_param_.video_data_param().stride())
	    interval = ui1(*prefetch_rng);
	else
	    interval = 1;
	int range = num_frames - (interval * (T - 1) + 1);
	CHECK_GE(range, 0) << "num_frames must be greater than T.";
	boost::uniform_int<> ui2(0, range);

	int start_frame_pos = ui2(*prefetch_rng);
//	int start_frame_pos = 0;
	capture.set(cv::CAP_PROP_POS_FRAMES, start_frame_pos);
    for(int t = 0; t < (T - 1) * interval + 1; ++t){
      // get a blob
      timer.Start();
      cv::Mat cv_img;
      //CHECK(capture.set(cv::CAP_PROP_POS_FRAMES, start_frame_pos + interval * t)) << "set fail";
      //LOG(INFO) << t << " : " <<capture.get(cv::CAP_PROP_POS_FRAMES);
      CHECK(capture.read(cv_img)) << "Could not read the video " << lines_[lines_id_];
      if(t % interval != 0) continue;
      if(!is_color)
    	  cv::cvtColor(cv_img, cv_img, CV_BGR2GRAY);
      cv::resize(cv_img,cv_img,cv::Size(new_width, new_height));
      read_time += timer.MicroSeconds();
      timer.Start();
      // Apply transformations (mirror, crop...) to the image
      std::vector<int> offset_vec(5,0);
      offset_vec[0] = t / interval;
      offset_vec[1] = item_id;
      int offset = batch->data_.offset(offset_vec);
      this->transformed_data_.set_cpu_data(prefetch_data + offset);
      this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
      trans_time += timer.MicroSeconds();
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
	  // We have reached the end. Restart from the first.
	  DLOG(INFO) << "Restarting data prefetching from start.";
	  lines_id_ = 0;
	  if (this->layer_param_.video_data_param().shuffle()) {
	    ShuffleVideos();
	  }
    }
  }
  caffe_set(batch->label_.count(), (Dtype)1, prefetch_label);
  caffe_set(batch_size, (Dtype)0, prefetch_label);
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);

}  // namespace caffe
#endif  // USE_OPENCV
