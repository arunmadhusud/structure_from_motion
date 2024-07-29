#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <map>
#include <cassert>


using namespace std;

const int IMAGE_DOWNSAMPLE = 1; // downsample the image to speed up processing
const double FOCAL_LENGTH = 2559.68 / IMAGE_DOWNSAMPLE; // focal length in pixels, after downsampling, guess from jpeg EXIF data
const int MIN_LANDMARK_SEEN = 3; // minimum number of camera views a 3d point (landmark) has to be seen to be used

struct SFM_Helper
{
    struct ImagePose
    {
        cv::Mat img; // down sampled image used for display
        cv::Mat desc; // feature descriptor
        std::vector<cv::KeyPoint> kp; // keypoint

        cv::Mat T; // 4x4 pose transformation matrix
        cv::Mat P; // 3x4 projection matrix

        // alias to clarify map usage below
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // seypoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
    };

    // 3D point
    struct Landmark
    {
        cv::Point3f pt;
        int seen = 0; // how many cameras have seen this point
        // add color for visualization
        cv::Vec3f color;
    };

    std::vector<ImagePose> img_pose;
    std::vector<Landmark> landmark;
};

int main(int argc, char **argv)
{
    SFM_Helper SFM;

    // Find matching features
    {
        using namespace cv;

        Ptr<AKAZE> feature = AKAZE::create();
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

        namedWindow("img", WINDOW_NORMAL);

        //take datapath from command line
        std::string dataPath = argv[1];
        std::string imgBasePath = dataPath + "images/";
        std::string imgPrefix = "P1180";
        std::string imgFileType = ".JPG";
        int imgStartIndex = 141;
        int imgEndIndex = 225;

        std::cout << "Reading images from " << imgBasePath << std::endl;
        for (int imgIndex = imgStartIndex; imgIndex <= imgEndIndex; imgIndex++)
        {
            std::string imgFullFilename = imgBasePath + imgPrefix + std::to_string(imgIndex) + imgFileType;
            cv::Mat img = cv::imread(imgFullFilename);
            if (!img.empty())
            {   
                SFM_Helper::ImagePose a;
                resize(img, img, img.size()/IMAGE_DOWNSAMPLE);
                a.img = img;
                cvtColor(img, img, COLOR_BGR2GRAY);

                feature->detect(img, a.kp);
                feature->compute(img, a.kp, a.desc);

                SFM.img_pose.emplace_back(a);
            }
            else {
                std::cout << "Image " << imgFullFilename << " is empty!" << std::endl;
            }
        }
        std::cout << "Number of images = " << SFM.img_pose.size() << std::endl;


        // Match features between all images
        for (size_t i=0; i < SFM.img_pose.size()-1; i++) {
            auto &img_pose_i = SFM.img_pose[i];
            // for (size_t j=i+1; j <= SFM.img_pose.size()-1; j++) {
            for (size_t j=i+1; j <= i+1; j++) {
                auto &img_pose_j = SFM.img_pose[j];
                vector<vector<DMatch>> matches;
                vector<Point2f> src, dst;
                vector<uchar> mask;
                vector<int> i_kp, j_kp;

                // 2 nearest neighbour match
                matcher->knnMatch(img_pose_i.desc, img_pose_j.desc, matches, 2);

                for (auto &m : matches) {
                    if(m[0].distance < 0.7*m[1].distance) {
                        src.push_back(img_pose_i.kp[m[0].queryIdx].pt);
                        dst.push_back(img_pose_j.kp[m[0].trainIdx].pt);

                        i_kp.push_back(m[0].queryIdx);
                        j_kp.push_back(m[0].trainIdx);
                    }
                }

                // Filter bad matches using fundamental matrix constraint
                findFundamentalMat(src, dst, FM_RANSAC, 3.0, 0.99, mask);

                Mat canvas = img_pose_i.img.clone();
                canvas.push_back(img_pose_j.img.clone());

                for (size_t k=0; k < mask.size(); k++) {
                    if (mask[k]) {
                        img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
                        img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

                        line(canvas, src[k], dst[k] + Point2f(0, img_pose_i.img.rows), Scalar(0, 0, 255), 2);
                    }
                }

                int good_matches = sum(mask)[0];
                assert(good_matches >= 8);

                cout << "Feature matching " << i << " " << j << " ==> " << good_matches << "/" << matches.size() << endl;

                resize(canvas, canvas, canvas.size()/2);

                imshow("img", canvas);
                waitKey(1);
            }
        }
    }

    // Recover motion between previous to current image and triangulate points
    {
        using namespace cv;

        // Setup camera matrix
        double cx = SFM.img_pose[0].img.size().width/2;
        double cy = SFM.img_pose[0].img.size().height/2;

        Point2d pp(cx, cy);

        Mat K = Mat::eye(3, 3, CV_64F);

        K.at<double>(0,0) = FOCAL_LENGTH;
        K.at<double>(1,1) = FOCAL_LENGTH;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        cout << endl << "initial camera matrix K " << endl << K << endl << endl;

        SFM.img_pose[0].T = Mat::eye(4, 4, CV_64F);
        SFM.img_pose[0].P = K*Mat::eye(3, 4, CV_64F);

        for (size_t i=0; i < SFM.img_pose.size() - 1; i++) {
            auto &prev = SFM.img_pose[i];
            auto &cur = SFM.img_pose[i+1];

            vector<Point2f> src, dst;
            vector<size_t> kp_used;

            for (size_t k=0; k < prev.kp.size(); k++) {
                if (prev.kp_match_exist(k, i+1)) {
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    src.push_back(prev.kp[k].pt);
                    dst.push_back(cur.kp[match_idx].pt);

                    kp_used.push_back(k);
                }
            }

            Mat mask;

            // NOTE: pose from dst to src
            Mat E = findEssentialMat(dst, src, FOCAL_LENGTH, pp, RANSAC, 0.999, 1.0, mask);
            Mat local_R, local_t;

            recoverPose(E, dst, src, local_R, local_t, FOCAL_LENGTH, pp, mask);

            // local tansform
            Mat T = Mat::eye(4, 4, CV_64F);
            local_R.copyTo(T(Range(0, 3), Range(0, 3)));
            local_t.copyTo(T(Range(0, 3), Range(3, 4)));

            // accumulate transform
            cur.T = prev.T*T;

            // make projection matrix
            Mat R = cur.T(Range(0, 3), Range(0, 3));
            Mat t = cur.T(Range(0, 3), Range(3, 4));

            Mat P(3, 4, CV_64F);

            P(Range(0, 3), Range(0, 3)) = R.t();
            P(Range(0, 3), Range(3, 4)) = -R.t()*t;
            P = K*P;

            cur.P = P;

            Mat points4D;
            triangulatePoints(prev.P, cur.P, src, dst, points4D);

            // Scale the new 3d points to be similar to the existing 3d points (landmark)
            // Use ratio of distance between pairing 3d points
            if (i > 0) {
                double scale = 0;
                int count = 0;

                Point3f prev_camera;

                prev_camera.x = prev.T.at<double>(0, 3);
                prev_camera.y = prev.T.at<double>(1, 3);
                prev_camera.z = prev.T.at<double>(2, 3);

                vector<Point3f> new_pts;
                vector<Point3f> existing_pts;

                for (size_t j=0; j < kp_used.size(); j++) {
                    size_t k = kp_used[j];
                    if (mask.at<uchar>(j) && prev.kp_match_exist(k, i+1) && prev.kp_3d_exist(k)) {
                        Point3f pt3d;

                        pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                        pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                        pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                        size_t idx = prev.kp_3d(k);
                        Point3f avg_landmark = SFM.landmark[idx].pt / (SFM.landmark[idx].seen - 1);
                        // color is average of all the colors
                    

                        new_pts.push_back(pt3d);
                        existing_pts.push_back(avg_landmark);
                    }
                }

                // ratio of distance for all possible point pairing
                // probably an over kill! can probably just pick N random pairs
                for (size_t j=0; j < new_pts.size()-1; j++) {
                    for (size_t k=j+1; k< new_pts.size(); k++) {
                        // double s = norm(existing_pts[j] - existing_pts[k]) / norm(new_pts[j] - new_pts[k]);
                        double s = norm(existing_pts[j] - prev_camera) / norm(new_pts[j] - prev_camera);

                        scale += s;
                        count++;
                    }
                }

                assert(count > 0);

                scale /= count;

                cout << "image " << (i+1) << " ==> " << i << " scale=" << scale << " count=" << count <<  endl;

                // apply scale and re-calculate T and P matrix
                local_t *= scale;

                // local tansform
                Mat T = Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                // accumulate transform
                cur.T = prev.T*T;

                // make projection ,matrix
                R = cur.T(Range(0, 3), Range(0, 3));
                t = cur.T(Range(0, 3), Range(3, 4));

                Mat P(3, 4, CV_64F);
                P(Range(0, 3), Range(0, 3)) = R.t();
                P(Range(0, 3), Range(3, 4)) = -R.t()*t;
                P = K*P;

                cur.P = P;

                triangulatePoints(prev.P, cur.P, src, dst, points4D);
            }

            // Find good triangulated points
            for (size_t j=0; j < kp_used.size(); j++) {
                if (mask.at<uchar>(j)) {
                    size_t k = kp_used[j];
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    Point3f pt3d;

                    pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                    pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                    pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                    cv::Vec3b color = cur.img.at<cv::Vec3b>(cur.kp[match_idx].pt);
                    // normalize color
                    cv::Vec3f colorf = cv::Vec3f(color[0], color[1], color[2]) / 255.0;

                    if (prev.kp_3d_exist(k)) {
                        // Found a match with an existing landmark
                        cur.kp_3d(match_idx) = prev.kp_3d(k);
                        SFM.landmark[prev.kp_3d(k)].pt += pt3d;
                        //update color
                        SFM.landmark[prev.kp_3d(k)].color += colorf;
                        SFM.landmark[cur.kp_3d(match_idx)].seen++;
                    } else {
                        // Add new 3d point
                        SFM_Helper::Landmark landmark;

                        landmark.pt = pt3d;
                        landmark.seen = 2;
                        landmark.color = colorf;

                        SFM.landmark.push_back(landmark);

                        prev.kp_3d(k) = SFM.landmark.size() - 1;
                        cur.kp_3d(match_idx) = SFM.landmark.size() - 1;
                    }
                }
            }
        }

        // Average out the landmark 3d position
        for (auto &l : SFM.landmark) {
            if (l.seen >= 3) {
                l.pt /= (l.seen - 1);
                l.color /= (l.seen - 1);
            }
        }   
        // write 3d points to file
        ofstream out("../3d_points.txt");
        for (auto &l : SFM.landmark) {
            out << l.pt.x << " " << l.pt.y << " " << l.pt.z << " " << l.color[0] << " " << l.color[1] << " " << l.color[2] << endl;
        }
        out.close();

    }

    return 0;
    
}
