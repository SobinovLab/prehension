#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <fstream>

#include "SharedQueue.h"
#include "Frame.h"


class FrameSavingBuffer
{
public:
	// dn has to end with separator '/'
	FrameSavingBuffer(
		const std::string cn, const std::string dn,
		const float fr,
		const int iw, const int ih);
	// without log of frames, just video filename
	FrameSavingBuffer(
		const std::string vn,
		const float fr,
		const int iw, const int ih);
	~FrameSavingBuffer();

	void add(const Frame& frame);

	std::atomic<bool> terminate = false;
	std::atomic<bool> stop_request = false;
	std::atomic<bool> log_timestamps = true;

	void stop_wait();  // will lock if none terminates or requests stop
	void stop_terminate();
private:
	SharedQueue<Frame> q;
	std::thread* looping_thread = nullptr;

	// these two only legacy for images
	const std::string camera_name;
	const std::string dirname;
	const float framerate;
	const int image_width;
	const int image_heigth;

	// these are used.
	std::string video_filename;
	std::string timestamps_filename;

	void start();  // runs at creation

	// internal loops
	void saving_loop_images();
	void saving_loop_videos();
	void saving_loop_videos_fill_in();

	// clear internal image buffer
	void release_clear_queue();

};

