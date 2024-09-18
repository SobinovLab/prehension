#include "Frame.h"

#ifndef MY_CHARBUF_SIZE
#define MY_CHARBUF_SIZE 2000
#endif // !MY_CHARBUF_SIZE

using namespace std;

Frame::Frame(cv::Mat im, const uint64_t gts, const uint64_t ts, const uint64_t fi) :
	image(im),
	global_time_stamp(gts),
	time_stamp(ts),
	frame_id(fi)
{

}

Frame::Frame(const Frame& ff) :
	image(ff.image),
	global_time_stamp(ff.global_time_stamp),
	time_stamp(ff.time_stamp),
	frame_id(ff.frame_id)
{
}

Frame::~Frame()
{

}

std::string Frame::image_filename(const std::string camera_name, const uint64_t time_start, const std::string ext)
{
	char buf[MY_CHARBUF_SIZE];
	sprintf(buf, "%s_%.4f_image%llu%s",
		camera_name.c_str(),
		(time_stamp - time_start) / 1.E6,
		frame_id,
		ext.c_str());
	return buf;
}

uint64_t Frame::get_time_stamp()
{
	return time_stamp;
}

uint64_t Frame::get_time_stamp_from(const uint64_t time_stamp)
{
	return time_stamp - time_stamp;
}

uint64_t Frame::get_global_time_stamp()
{
	return global_time_stamp;
}

uint64_t Frame::get_frame_id()
{
	return frame_id;
}


void Frame::save_image_cv2(const std::string filename)
{
	//cv::Mat cvmat(image->GetHeight() + image->GetYPadding(), image->GetWidth() + image->GetXPadding(), CV_8UC3);

	//Spinnaker::ImageProcessor ip;

	//Spinnaker::ImagePtr imageconv = ip.Convert(image, Spinnaker::PixelFormat_BGR8);

	//cvmat.data = (uchar*)imageconv->GetData();
	cv::imwrite(filename, image);
}

void Frame::save_image(const std::string filename)
{
	save_image_cv2(filename);
	// save_image_spinnaker(filename);
}

cv::Mat Frame::get_image()
{
	return image;
}

void Frame::release()
{
	//// if the camera was reset, the image buffer is wiped
	//try
	//{
	//	image->Release();
	//}
	//catch (const Spinnaker::Exception& me)
	//{
	//	string buf = "Image releasing error. Image " + buf + " Error: " + me.what();
	//	logError(buf.c_str());
	//}
	image.release();
}
