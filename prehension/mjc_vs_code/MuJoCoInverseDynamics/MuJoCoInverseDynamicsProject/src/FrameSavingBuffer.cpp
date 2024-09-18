#include "FrameSavingBuffer.h"

#ifndef MY_CHARBUF_SIZE
#define MY_CHARBUF_SIZE 2000
#endif // !MY_CHARBUF_SIZE

using namespace std;

FrameSavingBuffer::FrameSavingBuffer(
	const std::string cn, const std::string dn,
	const float fr,
	const int iw, const int ih) :
	camera_name(cn), dirname(dn), framerate(fr), image_width(iw), image_heigth(ih)
{
	timestamps_filename = dirname + camera_name + ".csv";
	video_filename = dirname + camera_name + ".mp4";
	start();
}

FrameSavingBuffer::FrameSavingBuffer(
	const std::string vn,
	const float fr,
	const int iw, const int ih) :
	video_filename(vn), framerate(fr), image_width(iw), image_heigth(ih)
{
	log_timestamps = false;
	start();
}

FrameSavingBuffer::~FrameSavingBuffer()
{
	stop_terminate();
}

void FrameSavingBuffer::add(const Frame& flirFrame)
{
	q.enqueue(flirFrame);
}

void FrameSavingBuffer::start()
{
	if (looping_thread) {
		printf("Error: Cannot start camera saving buffer. Terminate first.\n");
		return;
	}

	// IMAGES
	//looping_thread = new thread(&FrameSavingBuffer::saving_loop_images, this);

	// VIDEOS
	//looping_thread = new thread(&FrameSavingBuffer::saving_loop_videos, this);
	// VIDEOS - fil in empty
	looping_thread = new thread(&FrameSavingBuffer::saving_loop_videos_fill_in, this);
}

void FrameSavingBuffer::stop_wait()
{
	if (!looping_thread)
		return;

	looping_thread->join();
	looping_thread = nullptr;
}

void FrameSavingBuffer::stop_terminate()
{
	if (!looping_thread)
		return;

	terminate = true;
	looping_thread->join();
	looping_thread = nullptr;
}

void FrameSavingBuffer::saving_loop_images()
{
	// timestamps file
	ofstream fo;
	if (this->log_timestamps) {
		fo.open(timestamps_filename);
		fo << "Frame #, Time stamp (msec), Global Time Stamp (msec)\n";
	}

	// buffers
	string filename;
	uint64_t time_start;
	char buf_c[MY_CHARBUF_SIZE];
	bool first_frame = true;
	while (!terminate) {
		if (!q.empty()) {
			Frame ff(q.dequeue());

			if (first_frame) {
				first_frame = false;
				time_start = ff.get_time_stamp();
			}

			// SAVE
			filename = dirname + ff.image_filename(camera_name, time_start);

			ff.save_image(filename);

			// write to log
			if (this->log_timestamps) {
				fo << ff.get_frame_id();
				sprintf(buf_c, "%.3f", ff.get_time_stamp() / 1.E6);  // can get nanoseconds, but why?
				fo << ", " << buf_c;
				fo << ", " << ff.get_global_time_stamp();
				fo << "\n";
			}

			// release data
			ff.release();
		}
		else if (stop_request)
			break;
	}

	// clear up residual
	release_clear_queue();
}

void FrameSavingBuffer::saving_loop_videos()
{
	// timestamps file
	ofstream fo;
	if (this->log_timestamps) {
		fo.open(timestamps_filename);
		fo << "Frame #, Time stamp (msec), Global Time Stamp (msec)\n";
	}

	// video
	cv::VideoWriter video;
	//int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');     // Get Codec Type- Int form
	int fourcc = cv::VideoWriter::fourcc('h', '2', '6', '4');     // Get Codec Type- Int form
	// int fourcc = cv::VideoWriter::fourcc('h', '2', '6', '5');     // Get Codec Type- Int form -- not supported

	// cv::Size S(2048, 1536);
	cv::Size S(image_width, image_heigth);

	video.open(video_filename, fourcc, framerate, S, true);
	if (!video.isOpened()) {
		//logError("Video opening error.");
		return;
	}

	// buffers
	uint64_t time_start;
	char buf_c[MY_CHARBUF_SIZE];
	bool first_frame = true;
	while (!terminate) {
		if (!q.empty()) {
			Frame ff(q.dequeue());

			if (first_frame) {
				first_frame = false;
				time_start = ff.get_time_stamp();
			}

			// add to video
			video << ff.get_image();

			// write to log
			if (this->log_timestamps) {
				fo << ff.get_frame_id();
				sprintf(buf_c, "%.3f", ff.get_time_stamp() / 1.E6);  // can get nanoseconds, but why?
				fo << ", " << buf_c;
				fo << ", " << ff.get_global_time_stamp();
				fo << "\n";
			}

			// release data
			ff.release();
		}
		else if (stop_request)
			break;
	}

	// clear up residual
	release_clear_queue();
}


void FrameSavingBuffer::saving_loop_videos_fill_in()
{
	// timestamps file
	ofstream fo;
	if (this->log_timestamps) {
		fo.open(timestamps_filename);
		fo << "Frame #, Time stamp (msec), Global Time Stamp (msec)\n";
	}

	// video
	cv::VideoWriter video;
	//int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');     // Get Codec Type- Int form
	int fourcc = cv::VideoWriter::fourcc('h', '2', '6', '4');     // Get Codec Type- Int form
	// int fourcc = cv::VideoWriter::fourcc('h', '2', '6', '5');     // Get Codec Type- Int form -- not supported

	// cv::Size S(2048, 1536);
	cv::Size S(image_width, image_heigth);

	video.open(video_filename, fourcc, framerate, S, true);
	if (!video.isOpened()) {
		printf("Error: Video opening error.\n");
		return;
	}

	// buffers
	uint64_t time_start;
	uint64_t prev_frame_id;
	char buf_c[MY_CHARBUF_SIZE];
	bool first_frame = true;
	while (!terminate) {
		if (!q.empty()) {
			Frame ff(q.dequeue());

			if (first_frame) {
				first_frame = false;
				time_start = ff.get_time_stamp();
				prev_frame_id = ff.get_frame_id() - 1;  // overflow, but followed by ++ anyway
			}

			while (++prev_frame_id < ff.get_frame_id())
				video << ff.get_image();

			// add to video
			video << ff.get_image();

			// write to log
			if (this->log_timestamps) {
				fo << ff.get_frame_id();
				sprintf(buf_c, "%.3f", ff.get_time_stamp() / 1.E6);  // can get nanoseconds, but why?
				fo << ", " << buf_c;
				fo << ", " << ff.get_global_time_stamp();
				fo << "\n";
			}

			// release data
			ff.release();
		}
		else if (stop_request)
			break;
	}

	// clear up residual
	release_clear_queue();
}


void FrameSavingBuffer::release_clear_queue()
{
	while (!q.empty()) {
		q.dequeue().release();
	}
}
