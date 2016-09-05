/*
Note that the face detector is fastest when compiled with at least
SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
chip then you should enable at least SSE2 instructions.  If you are using
cmake to compile this program you can enable them by using one of the
following commands when you create the build project:
cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
This will set the appropriate compiler options for GCC, clang, Visual
Studio, or the Intel compiler.  If you are using another compiler then you
need to consult your compiler's manual to determine how to enable these
instructions.  Note that AVX is the fastest but requires a CPU from at least
2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <opencv2/viz/viz3d.hpp>
#include <opencv2/viz/types.hpp>
#include <opencv2/viz/vizcore.hpp>

#include <dlib/image_processing/full_object_detection.h>

#include <vector>

using namespace dlib;
using namespace cv;
using namespace viz;
using namespace std;

const char face_landmark[] = "shape_predictor_68_face_landmarks.dat";

namespace trek
{
    enum algorithm
    {
        SHAPE_DESCRIPTOR,
        OPTICAL_FLOW
    };

    //  Switch between detection and tracking when not finding face, or lose points for tracking
    bool detect = true;
    unsigned int algorithm = OPTICAL_FLOW;
    bool running = true;

    std::vector<Point3f> read_model_points(string filename)
    {
        std::vector<Point3f> points;

        float point_x, point_y, point_z;

        fstream fs;
        fs.open(filename);

        if (fs.is_open())
        {
            while (!fs.eof())
            {
                fs >> point_x;
                fs >> point_y;
                fs >> point_z;

                points.push_back(Point3f(point_x, point_y, point_z));

            }

            points.pop_back();

        }

        return points;
    }

    inline std::vector<Point2f> extract_frame_points(
        full_object_detection& d)
    {
        std::vector<Point2f> face_frame_points;

        // left most, then right most lips point
        face_frame_points.push_back(Point2f(d.part(48).x(), d.part(48).y()));
        face_frame_points.push_back(Point2f(d.part(54).x(), d.part(54).y()));

        // top of nose point
        face_frame_points.push_back(Point2f(d.part(30).x(), d.part(30).y()));

        // left bottom point of the nose, then right bottom point
        face_frame_points.push_back(Point2f(d.part(31).x(), d.part(31).y()));
        face_frame_points.push_back(Point2f(d.part(35).x(), d.part(35).y()));

        // outer most then inner most left eye point
        face_frame_points.push_back(Point2f(d.part(36).x(), d.part(36).y()));
        face_frame_points.push_back(Point2f(d.part(39).x(), d.part(39).y()));

        // outer most then inner most right eye point
        face_frame_points.push_back(Point2f(d.part(45).x(), d.part(45).y()));
        face_frame_points.push_back(Point2f(d.part(42).x(), d.part(42).y()));

        // outer most then inner most left eyebrow point
        face_frame_points.push_back(Point2f(d.part(17).x(), d.part(17).y()));
        face_frame_points.push_back(Point2f(d.part(21).x(), d.part(21).y()));

        // outer most then inner most right eyebrow point
        face_frame_points.push_back(Point2f(d.part(26).x(), d.part(26).y()));
        face_frame_points.push_back(Point2f(d.part(22).x(), d.part(22).y()));

        return face_frame_points;
    }

    inline void determine_head_pose(
        const std::vector<full_object_detection> dets,
        Matx33d& camera_matrix,
        Viz3d& pose_window,
        string& widget_ID)
    {
        Mat rvec(3, 1, DataType<double>::type);
        Mat tvec(3, 1, DataType<double>::type);

        if (dets.size() > 0)
        {
            std::vector<Point2f> face_frame_points;
            std::vector<Point3f> face_model_points;

            Mat dist_coeffs(4, 1, DataType<double>::type);
            dist_coeffs.at<double>(0) = 0;
            dist_coeffs.at<double>(1) = 0;
            dist_coeffs.at<double>(2) = 0;
            dist_coeffs.at<double>(3) = 0;

            face_model_points = read_model_points("face_model_points.txt");
            
            full_object_detection shape = dets[0];
            face_frame_points = extract_frame_points(shape);

            solvePnP(face_model_points, face_frame_points, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_EPNP);

            Mat rotation_mat;
            Rodrigues(rvec, rotation_mat);

            double _pm[16] =
            { 
                rotation_mat.at<double>(0, 0), rotation_mat.at<double>(0, 1), rotation_mat.at<double>(0, 2), tvec.at<double>(0),
                rotation_mat.at<double>(1, 0), rotation_mat.at<double>(1, 1), rotation_mat.at<double>(1, 2), tvec.at<double>(1),
                rotation_mat.at<double>(2, 0), rotation_mat.at<double>(2, 1), rotation_mat.at<double>(2, 2), tvec.at<double>(2),
                0,                             0,                             0,                             1
            };

            Matx44d P(_pm);

            Affine3d pose(P);

            pose_window.setWidgetPose(widget_ID, pose);
        }
    }

    inline void determine_head_pose(
        std::vector<Point2f> face_frame_points,
        Matx33d& camera_matrix,
        Viz3d& pose_window,
        string& widget_ID)
    {
        Mat rvec(3, 1, DataType<double>::type);
        Mat tvec(3, 1, DataType<double>::type);

        std::vector<Point3f> face_model_points;

        Mat dist_coeffs(4, 1, DataType<double>::type);
        dist_coeffs.at<double>(0) = 0;
        dist_coeffs.at<double>(1) = 0;
        dist_coeffs.at<double>(2) = 0;
        dist_coeffs.at<double>(3) = 0;

        face_model_points = read_model_points("face_model_points.txt");

        solvePnP(face_model_points, face_frame_points, camera_matrix, dist_coeffs, rvec, tvec, false, SOLVEPNP_EPNP);

        Mat rotation_mat;
        Rodrigues(rvec, rotation_mat);

        double _pm[16] =
        {
            rotation_mat.at<double>(0, 0), rotation_mat.at<double>(0, 1), rotation_mat.at<double>(0, 2), tvec.at<double>(0),
            rotation_mat.at<double>(1, 0), rotation_mat.at<double>(1, 1), rotation_mat.at<double>(1, 2), tvec.at<double>(1),
            rotation_mat.at<double>(2, 0), rotation_mat.at<double>(2, 1), rotation_mat.at<double>(2, 2), tvec.at<double>(2),
            0, 0, 0, 1
        };

        Matx44d P(_pm);

        Affine3d pose(P);

        pose_window.setWidgetPose(widget_ID, pose);
    }

    inline std::vector<image_window::overlay_line> render_face_mask(
        const std::vector<full_object_detection>& dets,
        const rgb_pixel color = rgb_pixel(0, 255, 0)
        )
    {
        std::vector<image_window::overlay_line> lines;

        std::vector<Point2f> face_frame_points;

        for (unsigned long i = 0; i < dets.size(); ++i)
        {
            DLIB_CASSERT(dets[i].num_parts() == 68,
                "\t std::vector<image_window::overlay_line> render_face_detections()"
                << "\n\t Invalid inputs were given to this function. "
                << "\n\t dets[" << i << "].num_parts():  " << dets[i].num_parts()
                );

            const full_object_detection& d = dets[i];

            // Around Chin. Ear to Ear
            for (unsigned long i = 1; i <= 16; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));

            // Line on top of nose
            for (unsigned long i = 28; i <= 30; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));

            // left eyebrow
            for (unsigned long i = 18; i <= 21; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));

            // Right eyebrow
            for (unsigned long i = 23; i <= 26; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));

            // Bottom part of the nose
            for (unsigned long i = 31; i <= 35; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));
            // Line from the nose to the bottom part above
            lines.push_back(image_window::overlay_line(d.part(30), d.part(35), color));

            // Left eye
            for (unsigned long i = 37; i <= 41; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));
            lines.push_back(image_window::overlay_line(d.part(36), d.part(41), color));

            // Right eye
            for (unsigned long i = 43; i <= 47; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));
            lines.push_back(image_window::overlay_line(d.part(42), d.part(47), color));

            // Lips outer part
            for (unsigned long i = 49; i <= 59; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));
            lines.push_back(image_window::overlay_line(d.part(48), d.part(59), color));

            // Lips inside part
            for (unsigned long i = 61; i <= 67; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i - 1), color));
            lines.push_back(image_window::overlay_line(d.part(60), d.part(67), color));
        }
        return lines;
    }

    // ----------------------------------------------------------------------------------------

    inline std::vector<image_window::overlay_line> render_face_mask(
        const full_object_detection& det,
        const rgb_pixel color = rgb_pixel(0, 255, 0)
        )
    {
        std::vector<full_object_detection> dets;
        dets.push_back(det);
        return render_face_detections(dets, color);
    }

    // ----------------------------------------------------------------------------------------
}

using namespace trek;

int main()
{
    try
    {
        VideoCapture cap(0);
        if (!cap.isOpened())
        {
            cerr << "Unable to connect to camera" << endl;
            return 1;
        }

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize(face_landmark) >> pose_model;

        Viz3d pose_window("Head Pose Estimation");

        Mesh head_pose_mesh = Mesh::load("HUMAN HEAD.ply");
        WMesh head_pose_widget(head_pose_mesh);
        
        string widget_ID = "Head Widget";
        pose_window.showWidget(widget_ID, head_pose_widget);
        
        double viewer_pose_arr[16] =
        {
            1,  0,  0,  0,
            0,  1,  0,  0,
            0,  0,  1,  0,
            0,  0,  0,  1
        };
        Matx44d viewer_pose_mat(viewer_pose_arr);
        pose_window.setViewerPose(Affine3d(viewer_pose_mat));

        //  Optical Flow parameters
        TermCriteria term_crit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        Size sub_pix_win_size(17, 17), win_size(31, 31);

        Mat frame, gray, previous_gray;
        std::vector<Point2f> points, previous_points, corners, previous_corners;

        image_window win;
        win.set_title("Face Alignment");

        // Grab and process frames until the main window is closed by the user.
        while (running)
        {
            // Grab a frame
            cap >> frame;

            if (frame.empty())
                break;

            cvtColor(frame, gray, COLOR_BGR2GRAY);

            //  Camera calibration matrix
            int focal_length = max(frame.rows, frame.cols);
            double principal_point_x = frame.cols / 2;
            double principal_point_y = frame.rows / 2;
            Size window_size(frame.cols, frame.rows);

            double _cm[9] =
            { focal_length, 0, principal_point_x,
            0, focal_length, principal_point_y,
            0, 0, 1
            };
            Matx33d camera_mat(_cm);

            pose_window.setWindowSize(window_size);

            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(frame);
            
            if (algorithm == SHAPE_DESCRIPTOR)
            {
                // Face Detection
                std::vector<dlib::rectangle> faces = detector(cimg);

                // Find the 2D pose of each face
                std::vector<full_object_detection> shapes;

                for_each(faces.begin(), faces.end(), [&pose_model, &shapes, &cimg](dlib::rectangle& face)
                {
                full_object_detection shape = pose_model(cimg, face);

                shapes.push_back(shape);
                });

                //  3D Pose Estimation
                determine_head_pose(shapes, camera_mat, pose_window, widget_ID);
                pose_window.spinOnce();

                // Display it all on the screen
                win.clear_overlay();
                win.set_image(cimg);
                win.add_overlay(render_face_mask(shapes));
            }
            else if (algorithm == OPTICAL_FLOW)
            {
                if (detect)
                {
                    // Face Detection
                    std::vector<dlib::rectangle> faces = detector(cimg);

                    if (!faces.empty())
                    {
                        detect = false;

                        previous_points = extract_frame_points(pose_model(cimg, faces[0]));
                    }
                }

                if (!previous_points.empty())
                {
                    previous_corners = std::vector<Point2f>(previous_points);

                    cornerSubPix(gray, previous_corners, sub_pix_win_size, Size(-1, -1), term_crit);

                    //  Optical Flow Head Tracking
                    std::vector<uchar> status;
                    std::vector<float> err;

                    if (previous_gray.empty())
                        gray.copyTo(previous_gray);

                    calcOpticalFlowPyrLK(previous_gray, gray, previous_corners, corners, status, err, win_size, 3, term_crit, 0, 0.001);

                    Mat homography = findHomography(previous_corners, corners);

                    perspectiveTransform(previous_points, points, homography);

                    for_each(points.begin(), points.end(), [&](Point2f& point) {
                        circle(frame, point, 3, Scalar(0, 255, 0)); });

                    //  3D Pose Estimation
                    determine_head_pose(points, camera_mat, pose_window, widget_ID);
                    pose_window.spinOnce();

                    std::swap(previous_points, points);
                    cv::swap(previous_gray, gray);
                }

                namedWindow("Optical Flow");
                cv::imshow("Optical Flow", frame);
            }

            char c = (char)waitKey(60);

            switch (c)
            {
            case 'd':
                detect = true;
                break;
            case 'a':
                algorithm = algorithm == SHAPE_DESCRIPTOR ? OPTICAL_FLOW : SHAPE_DESCRIPTOR;
                break;
            case 'x':
                running = false;
                break;
            }
        }
    }
    catch (serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
    }
}

