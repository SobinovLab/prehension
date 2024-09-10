#pragma once

#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>

class Frame
{
public:
	Frame(cv::Mat im, const uint64_t gts, const uint64_t ts, const uint64_t fi);
	Frame(const Frame& ff);
	~Frame();

	// image filename
	std::string image_filename(const std::string camera_name, const uint64_t time_start, const std::string ext = ".jpeg");

	// image time stamp
	uint64_t get_time_stamp();
	uint64_t get_time_stamp_from(const uint64_t time_stamp);
	uint64_t get_global_time_stamp();
	uint64_t get_frame_id();

	// save to file
	// full filename
	void save_image_cv2(const std::string filename);
	void save_image(const std::string filename);

	// get for video
	cv::Mat get_image();

	// call when done using
	void release();

private:
	Frame() = delete;

	cv::Mat image;
	const uint64_t global_time_stamp;
	const uint64_t time_stamp;
	const uint64_t frame_id;
};

