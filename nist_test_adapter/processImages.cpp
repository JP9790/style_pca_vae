#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "voronoi/src/voronoi.h"
#include "connectedComponents/connectedComponents.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

class VoronoiIterator {
public:
  VoronoiIterator() {}

  void init(const cv::Mat1b & query, std::string implementation_name_, bool crop_img_before_)
  {
    _implementation_name = implementation_name_;
    _crop_img_before = crop_img_before_;
    _first_img = query.clone();
    VoronoiThinner::copy_bounding_box_plusone(query, _first_img, true);
    _curr_skel = _first_img.clone();
    _curr_iter = 1;
    _nframes = 0;
  }

  inline cv::Mat1b first_img() {
    return _first_img.clone();
  }

  cv::Mat1b current_skel() const {
    return _thinner.get_skeleton().clone();
  }

  inline cv::Mat1b contour_brighter(const cv::Mat1b & img) {
    _contour_viz.from_image_C4(img);
    cv::Mat1b ans;
    _contour_viz.copyTo(ans);
    return ans;
  }

  inline cv::Mat3b contour_color(const cv::Mat1b & img) {
    _contour_viz.from_image_C4(img);
    return _contour_viz.illus().clone();
  }

  bool iter() {
    ++_nframes;
    bool reuse = (_implementation_name != IMPL_MORPH);
    bool success = false;
    if (reuse)
      success = _thinner.thin(_curr_skel, _implementation_name, false, 1);
    else
      success = _thinner.thin(_first_img, _implementation_name, false, _nframes);
    _thinner.get_skeleton().copyTo(_curr_skel);
    return success;
  }

  inline bool has_converged() const { return _thinner.has_converged(); }
  inline int cols() const { return _first_img.cols; }
  inline int rows() const { return _first_img.rows; }

  std::string _implementation_name;
  bool _crop_img_before;
  int _nframes;
  int _curr_iter;
  cv::Mat1b _first_img;
  cv::Mat1b _curr_skel;
  VoronoiThinner _thinner;
  ImageContour _contour_viz;
};

void writeImage( string fn, Mat binaryImage)
{
  ofstream outputFile;
  outputFile.open( fn );

  for (int r = 0; r < binaryImage.rows; r++){
    for (int c = 0; c < binaryImage.cols; c++){
      int pixel = binaryImage.at<uchar>(r,c);
      outputFile << pixel << '\t';
    }
    outputFile << endl;
  }
  outputFile.close();
}

void voronoiThinImage(const Mat& input_image, Mat& result, string thinning_method)
{
  const cv::Mat1b & query = input_image.clone();
  VoronoiThinner thinner;

  bool success = thinner.thin( query, thinning_method, false, 1);
  thinner.get_skeleton().copyTo( result );
}

void processImage(const string& input_path, const string& output_dir, const string& image_name)
{
  cout << "Processing: " << input_path << endl;
  
  // Read image
  Mat input_image = imread(input_path, IMREAD_GRAYSCALE);
  if (input_image.empty()) {
    cerr << "Could not read image: " << input_path << endl;
    return;
  }

  // Save original
  string base_name = output_dir + "/" + image_name;
  imwrite(base_name + "-input.png", input_image);
  writeImage(base_name + "-input.txt", input_image);

  // Find connected components
  std::vector<ConnectedComponent> components;
  findCC4(input_image.clone(), components);
  int nrcomponents_org4 = components.size();
  findCC8(input_image.clone(), components);
  int nrcomponents_org8 = components.size();
  int orgnrpoints = cv::countNonZero(input_image);

  // Adaptive thresholding
  Mat thresholded;
  int thr = 0;
  int stepsize = 25;
  bool done = false;
  int nrcomponents4, nrcomponents8;
  
  while(!done) {
    cv::threshold(input_image.clone(), thresholded, thr, 255, CV_THRESH_BINARY);

    findCC4(thresholded.clone(), components);
    nrcomponents4 = components.size();
    findCC8(thresholded.clone(), components);
    nrcomponents8 = components.size();

    int nrpoints = cv::countNonZero(thresholded);
    
    if (nrpoints < 0.5 * orgnrpoints)
      done = true;
    else if ((nrcomponents4 != nrcomponents_org4) || (nrcomponents8 != nrcomponents_org8) || (thr >= 250))
      done = true;
    else
      thr += stepsize;
  }
  
  thr = max(0, thr - stepsize);
  cv::threshold(input_image.clone(), thresholded, thr, 255, CV_THRESH_BINARY);

  // Save thresholded
  imwrite(base_name + "-thresholded.png", thresholded);
  writeImage(base_name + "-thresholded.txt", thresholded);

  // Apply thinning
  Mat thinned;
  voronoiThinImage(thresholded, thinned, "zhang_suen");

  // Save thinned
  imwrite(base_name + "-thinned.png", thinned);
  writeImage(base_name + "-thinned.txt", thinned);
}

int main(int argc, char* argv[])
{
  if (argc < 3) {
    cout << "Usage: ./processImages <input-directory> <output-directory>" << endl;
    return 1;
  }

  string input_dir = argv[1];
  string output_dir = argv[2];

  cout << "Input directory: " << input_dir << endl;
  cout << "Output directory: " << output_dir << endl;

  // Process all PNG files in input directory and subdirectories
  int img_count = 0;
  for (const auto& entry : fs::recursive_directory_iterator(input_dir)) {
    if (entry.is_regular_file()) {
      string path = entry.path().string();
      if (path.substr(path.find_last_of(".") + 1) == "png") {
        string filename = entry.path().filename().string();
        string stem = entry.path().stem().string();
        processImage(path, output_dir, stem);
        img_count++;
      }
    }
  }

  cout << "Processed " << img_count << " images." << endl;
  return 0;
}
