/**
 * @file geometry.cpp
 * @brief 眼动监测项目的相关函数实现
 * @author LiuXin 
 * @date 2025-05-16
 */
#include "geometry.h"
#include <stdexcept>

// 定义XYZ轴常量
const Geometry_Vector X_AXIS{1.0, 0.0, 0.0};
const Geometry_Vector Y_AXIS{0.0, 1.0, 0.0};
const Geometry_Vector Z_AXIS{0.0, 0.0, 1.0};

/** @struct Geometry_Point
 * @brief 定义一个点结构体
 */
Geometry_Point::Geometry_Point() : x(0.0), y(0.0), z(0.0) {}
Geometry_Point::Geometry_Point(double x, double y, double z) : x(x), y(y), z(z) {}
void Geometry_Point::Assign(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
/** @brief 将cv::Mat转换为Geometry_Point */
Geometry_Point Geometry_Point::Mat2Point(const cv::Mat& mat)
{
    Geometry_Point point;
    //使得对行向量和列向量都能识别
    if (mat.rows == 1)
    {
        point = Geometry_Point(mat.at<double>(0,0),mat.at<double>(0,1),mat.at<double>(0,2));
    }
    else if (mat.cols == 1)
    {
        point = Geometry_Point(mat.at<double>(0,0),mat.at<double>(1,0),mat.at<double>(2,0));
    }
    return point;

}
/** @brief 将Geometry_Point转换为cv::Mat */
cv::Mat Geometry_Point::Point2Mat(const Geometry_Point& p)
{
    cv::Mat mat(3,1,CV_64FC1);
    mat.at<double>(0,0)=p.x;
    mat.at<double>(1,0)=p.y;
    mat.at<double>(2,0)=p.z;
    return mat;
}

/** @struct Geometry_Vector
 * @brief 描述一个向量结构体
 */
Geometry_Vector::Geometry_Vector() : x(0.0), y(0.0), z(0.0) {}
Geometry_Vector::Geometry_Vector(double x, double y, double z) 
    : x(x), y(y), z(z) {}
void Geometry_Vector::Assign(double x, double y, double z) {
    this->x = x;
    this->y = y;
    this->z = z;
}
/** @brief 向量的模长 */
double Geometry_Vector::Magnitude() const {
    return std::sqrt(x * x + y * y + z * z);
}
/**  @brief 向量的点积 */
double Geometry_Vector::DotProduct(const Geometry_Vector& v) const {
    return x * v.x + y * v.y + z * v.z;
}
/** @brief 向量的叉积 */
Geometry_Vector Geometry_Vector::CrossProduct(const Geometry_Vector& v) const 
{
    return Geometry_Vector(
        y * v.z - z * v.y,
        z * v.x - x * v.z,
        x * v.y - y * v.x
    );
}
/** @brief 向量加法 */
Geometry_Vector Geometry_Vector::operator+(const Geometry_Vector& v) const 
{
    return Geometry_Vector(x + v.x, y + v.y, z + v.z);
}

/** @brief 向量减法 */
Geometry_Vector Geometry_Vector::operator-(const Geometry_Vector& v) const 
{
    return Geometry_Vector(x - v.x, y - v.y, z - v.z);
}
/** @brief 向量数乘，向量*标量 */
Geometry_Vector Geometry_Vector::operator*(double scalar) const 
{
    return Geometry_Vector(x * scalar, y * scalar, z * scalar);
}
/** @brief 向量数乘，标量*向量 */
Geometry_Vector operator*(double scalar, const Geometry_Vector& v) 
{
    return Geometry_Vector(scalar * v.x, scalar * v.y, scalar * v.z);
}

/** @brief Geometry_Line类相关函数  */
Geometry_Line::Geometry_Line() : start(), end() {}
Geometry_Line::Geometry_Line(const Geometry_Point& s, const Geometry_Point& e)
    : start(s), end(e) {}
void Geometry_Line::Assign(const Geometry_Point& s, const Geometry_Point& e) {
    start = s;
    end = e;
    direction = Geometry_Vector(end.x - start.x, end.y - start.y, end.z - start.z);
    double direction_length = direction.Magnitude();
    direction.Assign(direction.x / direction_length, direction.y / direction_length, direction.z / direction_length);
}
void Geometry_Line::Assign(const Geometry_Point& s, const Geometry_Vector& v)
{
    start = s;
    double v_length = v.Magnitude();
    direction = Geometry_Vector(v.x/v_length,v.y/v_length,v.z/v_length);
    end = Geometry_Point(start.x+direction.x,start.y+direction.y,start.z+direction.z);
}


/** @brief 角度计算相关函数 */

//向量与x轴之间的夹角，记作phi, 单位：弧度
double AnglePhi(const Geometry_Vector& v)
{
    double v_length=v.Magnitude();
    if (v_length == 0.0) return 0.0;
    double phi = std::acos(v.DotProduct(X_AXIS) / v_length);
    return phi;
}
//向量在yz平面上的投影向量与z轴的夹角，记作theta，单位弧度
double AngleTheta(const Geometry_Vector& v)
{
    double v_length=v.Magnitude();
    if (v_length == 0.0) return 0.0;
    Geometry_Vector v_projection = v-(v.DotProduct(X_AXIS))*X_AXIS;
    double theta = std::acos(v_projection.DotProduct(Z_AXIS) / v_length);
    return theta;
}
//两向量的phi角之差，记作alpha，单位：弧度
double AngleAlpha(const Geometry_Vector& v1, const Geometry_Vector& v2)
{
    double alpha = AnglePhi(v1) - AnglePhi(v2);
    return alpha;
}
//两向量的theta角之差，记作beta，单位：弧度
double AngleBeta(const Geometry_Vector& v1, const Geometry_Vector& v2)
{
    double beta = AngleTheta(v1) - AngleTheta(v2);
    return beta;
}
double RadianToAngle(double radian, int decimals)
{
    double angle=std::round(radian * 180.0 / M_PI * pow(10, decimals)) / pow(10, decimals);
    return angle;
}


//kappa角定义视线

KappaAngle::KappaAngle(): alpha(0.0), beta(0.0),angle(0.0),alpha_left(0.0),beta_left(0.0),angle_left(0.0){}
KappaAngle::KappaAngle(double alpha, double beta,double angle,double alpha_left, double beta_left,double angle_left): 
    alpha(alpha), beta(beta),angle(angle),alpha_left(alpha_left),beta_left(beta_left),angle_left(angle_left) {}
KappaAngle::KappaAngle(double alpha, double beta,double angle): alpha(alpha), beta(beta),angle(angle) {}
void KappaAngle::Assign(double alpha, double beta,double angle)
{
    this->alpha = alpha;
    this->beta = beta;
    this->angle = angle;
}
void KappaAngle::Assign(double alpha, double beta,double angle,double alpha_left, double beta_left,double angle_left)
{
    this->alpha = alpha;
    this->beta = beta;
    this->angle = angle;
    this->alpha_left = alpha_left;
    this->beta_left = beta_left;
    this->angle_left = angle_left;
}


//GazePoint相关实现
void GazePoint::Assign(double gaze_point_x_physical,double gaze_point_y_physical,double gaze_point_z_physical)
{
    this->gaze_point_x_physical = gaze_point_x_physical;
    this->gaze_point_y_physical = gaze_point_y_physical;
    this->gaze_point_z_physical = gaze_point_z_physical;
}
GazePoint::GazePoint(): gaze_point_x_physical(0.0),gaze_point_y_physical(0.0),gaze_point_z_physical(0.0){}

/////////////////计算凝视点////////////////////////
// 默认构造函数：显式初始化所有字段
Data::Data()
    : o(0.0, 0.0, 0.0), 
      p(0.0, 0.0, 0.0), 
      ol(0.0, 0.0, 0.0), 
      pl(0.0, 0.0, 0.0), 
      F(0.0, 0.0, 0.0),
      E(0.0, 0.0, 0.0) {}

// 简化构造函数（仅右眼数据）
Data::Data(const Geometry_Point& o, const Geometry_Point& p, 
           const Geometry_Point& F, const Geometry_Point& E)
    : o(o), p(p),  
      ol(), pl(), 
      F(F), E(E) {}// 显式调用默认构造

// 完整构造函数（修正参数顺序）
Data::Data(const Geometry_Point& o, const Geometry_Point& p,
           const Geometry_Point& ol, const Geometry_Point& pl, 
           const Geometry_Point& F, const Geometry_Point& E)
    : o(o), p(p), ol(ol), pl(pl), F(F), E(E) {}
void Data::Assign(const Geometry_Point& o, const Geometry_Point& p, const Geometry_Point& F, const Geometry_Point& E)
{
    this->o = o;
    this->p = p;
    this->F = F;
    this->E = E;
}
void Data::Assign(const Geometry_Point& o, const Geometry_Point& p,
                  const Geometry_Point& ol, const Geometry_Point& pl, 
                  const Geometry_Point& F, const Geometry_Point& E)
{
    this->o = o;
    this->p = p;
    this->ol = ol;
    this->pl = pl;
    this->F = F;
    this->E = E;
}

KappaAngle CalculateKappaAngle(const Data& data)
{
    KappaAngle kappa_angle;
    Geometry_Vector op,pF;
    op.Assign(data.p.x-data.o.x,data.p.y-data.o.y,data.p.z-data.o.z);
    pF.Assign(data.F.x-data.p.x,data.F.y-data.p.y,data.F.z-data.p.z);
    double angle_right = std::acos(op.DotProduct(pF));
    Geometry_Vector op_left,pF_left;
    op_left.Assign(data.pl.x-data.ol.x,data.pl.y-data.ol.y,data.pl.z-data.ol.z);
    pF_left.Assign(data.F.x-data.pl.x,data.F.y-data.pl.y,data.F.z-data.pl.z);
    double angle_left = std::acos(op_left.DotProduct(pF_left));
    kappa_angle.Assign(AngleAlpha(op,pF),AngleBeta(op,pF),angle_right,
                        AngleAlpha(op_left,pF_left),AngleBeta(op_left,pF_left),angle_left);
    return kappa_angle;
}
KappaAngle CalculateKappaAngle(const DataList& data)
{
    KappaAngle kappa_angle;
    double alpha_sum_right = 0.0;
    double beta_sum_right = 0.0;
    double angle_sum_right = 0.0;
    double alpha_sum_left = 0.0;
    double beta_sum_left = 0.0;
    double angle_sum_left = 0.0;
    for(const Data& data : data)
    {
        KappaAngle kappa_angle_temp = CalculateKappaAngle(data);
        alpha_sum_right += kappa_angle_temp.alpha;
        beta_sum_right += kappa_angle_temp.beta;
        angle_sum_right += kappa_angle_temp.angle;
        alpha_sum_left += kappa_angle_temp.alpha_left;
        beta_sum_left += kappa_angle_temp.beta_left;
        angle_sum_left += kappa_angle_temp.angle;
    }
    double alpha_ave=alpha_sum_right/data.size();
    double beta_ave=beta_sum_right/data.size();
    double angle_ave=angle_sum_right/data.size();
    double alpha_ave_left=alpha_sum_left/data.size();
    double beta_ave_left=beta_sum_left/data.size();
    double angle_ave_left=angle_sum_left/data.size();
    kappa_angle.Assign(alpha_ave,beta_ave,angle_ave,
                        alpha_ave_left,beta_ave_left,angle_ave_left);
    return kappa_angle;
}
KappaAngle CalculateKappaAngle(const std::vector<DataList>& data)
{
    KappaAngle kappa_angle;
    int size=data.size();
    double alpha_sum_right = 0.0;
    double beta_sum_right = 0.0;
    double angle_sum_right = 0.0;
    double alpha_sum_left = 0.0;
    double beta_sum_left = 0.0;
    double angle_sum_left = 0.0;
    for(int i=0;i<size;i++)
    {
        KappaAngle kappa_angle_temp = CalculateKappaAngle(data[i]);
        alpha_sum_right += kappa_angle_temp.alpha;
        beta_sum_right += kappa_angle_temp.beta;
        angle_sum_right += kappa_angle_temp.angle;
        alpha_sum_left += kappa_angle_temp.alpha_left;
        beta_sum_left += kappa_angle_temp.beta_left;
        angle_sum_left += kappa_angle_temp.angle;
    }
    double alpha_ave_right = alpha_sum_right/size;
    double beta_ave_right = beta_sum_right/size;
    double angle_ave_right = angle_sum_right/size;
    double alpha_ave_left = alpha_sum_left/size;
    double beta_ave_left = beta_sum_left/size;
    double angle_ave_left = angle_sum_left/size;
    kappa_angle.Assign(alpha_ave_right,beta_ave_right,angle_ave_right,
                        alpha_ave_left,beta_ave_left,angle_ave_left);
    return kappa_angle;
}


void SaveKappaAngleToCSV(const std::string& filename, const KappaAngle& kappa_angle) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("无法创建文件: " + filename);
    }
    
    // 写入CSV标题行
    file << "alpha,beta,angle,alpha_left,beta_left,angle_left\n";
    
    // 设置浮点数精度（保留6位小数）
    file << std::fixed << std::setprecision(6);
    
    // 写入数据行
    file << kappa_angle.alpha << ","
         << kappa_angle.beta << ","
         << kappa_angle.angle << ","
         << kappa_angle.alpha_left << ","
         << kappa_angle.beta_left << ","
         << kappa_angle.angle_left << "\n";
}

KappaAngle LoadKappaAngleFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("无法打开文件: " + filename);

    KappaAngle kappa;
    std::string line;
    int line_num = 0;

    try {
        // 读取标题行
        std::getline(file, line);
        line_num++;
        
        // 验证标题格式
        if (line != "alpha,beta,angle,alpha_left,beta_left,angle_left") {
            throw std::runtime_error("CSV文件标题格式不匹配");
        }

        // 读取数据行
        if (std::getline(file, line)) {
            line_num++;
            std::stringstream ss(line);
            std::vector<std::string> tokens;
            std::string token;
            
            // 分割token
            while (std::getline(ss, token, ',')) 
                tokens.push_back(token);

            // 列数验证
            if (tokens.size() != 6) {
                throw std::runtime_error("数据列数应为6，实际为" + 
                                       std::to_string(tokens.size()));
            }

            // 数值解析lambda
            auto parse = [](const std::string& s) -> double {
                try { return std::stod(s); } 
                catch (...) { throw std::runtime_error("无效数值: " + s); }
            };

            // 解析各字段
            kappa.alpha = parse(tokens[0]);
            kappa.beta = parse(tokens[1]);
            kappa.angle = parse(tokens[2]);
            kappa.alpha_left = parse(tokens[3]);
            kappa.beta_left = parse(tokens[4]);
            kappa.angle_left = parse(tokens[5]);
        } else {
            throw std::runtime_error("文件缺少数据行");
        }
    } 
    catch (const std::exception& e) {
        throw std::runtime_error("[第" + std::to_string(line_num) + 
                                "行错误] " + e.what());
    }
    
    return kappa;
}
void SaveToCSV(const std::string& filename, const std::vector<Data>& data_list, bool include_left) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("无法创建文件: " + filename);
    }
    //如果不包含左眼
    if(!include_left)
    {

        // 写入CSV标题行
        file << "o_x,o_y,o_z,p_x,p_y,p_z,F_x,F_y,F_z,E_x,E_y,E_z\n";

        // 设置浮点数精度（保留6位小数）
        file << std::fixed << std::setprecision(6);

        // 写入数据行
        for (const auto& data : data_list) {
            file << data.o.x << "," << data.o.y << "," << data.o.z << ","
                << data.p.x << "," << data.p.y << "," << data.p.z << ","
                << data.F.x << "," << data.F.y << "," << data.F.z << ","
                <<data.E.x << "," << data.E.y << "," << data.E.z << "\n";
        }        
    }else
    {
        // 标题行（包含左右眼所有字段）
        file << "o_x,o_y,o_z,p_x,p_y,p_z,"
            << "ol_x,ol_y,ol_z,pl_x,pl_y,pl_z,"
            <<"F_x,F_y,F_z,E_x,E_y,E_z\n";

        file << std::fixed << std::setprecision(6);
        for (const auto& data : data_list) {
            file << data.o.x << "," << data.o.y << "," << data.o.z << ","
                << data.p.x << "," << data.p.y << "," << data.p.z << ","
                << data.ol.x << "," << data.ol.y << "," << data.ol.z << ","
                << data.pl.x << "," << data.pl.y << "," << data.pl.z << ","
                << data.F.x << "," << data.F.y << "," << data.F.z << ","
                << data.E.x << "," << data.E.y << "," << data.E.z << "\n";
        }
    }

}
DataList LoadFromCSV(const std::string& filename) {
    DataList data_list;
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("无法打开文件: " + filename);

    std::string line;
    int line_num = 0;
    bool has_left_eye_data = false;

    // 读取标题行判断数据格式
    std::getline(file, line);
    line_num++;
    
    // 判断是否包含左眼数据
    has_left_eye_data = (line.find("ol_x") != std::string::npos);

    // 根据标题行验证列数
    std::vector<std::string> header_tokens;
    std::stringstream header_ss(line);
    std::string header_token;
    while (std::getline(header_ss, header_token, ',')) {
        header_tokens.push_back(header_token);
    }

    const int expected_columns = has_left_eye_data ? 18 : 12;
    if (header_tokens.size() != expected_columns) {
        throw std::runtime_error("CSV标题列数不符，应为" + 
                                std::to_string(expected_columns) + 
                                "，实际为" + 
                                std::to_string(header_tokens.size()));
    }

    // 数据行处理
    while (std::getline(file, line)) {
        line_num++;
        std::stringstream ss(line);
        Data data;
        std::vector<std::string> tokens;
        std::string token;

        try {
            // 解析token
            while (std::getline(ss, token, ',')) 
                tokens.push_back(token);

            // 列数验证
            if (tokens.size() != expected_columns) 
                throw std::runtime_error("列数不符，应为" + 
                                        std::to_string(expected_columns) + 
                                        "，实际为" + 
                                        std::to_string(tokens.size()));

            // 数值解析lambda
            auto parse = [](const std::string& s) -> double {
                try { return std::stod(s); } 
                catch (...) { throw std::runtime_error("无效数值: " + s); }
            };

            // 公共字段解析
            data.o.x = parse(tokens[0]);
            data.o.y = parse(tokens[1]);
            data.o.z = parse(tokens[2]);
            data.p.x = parse(tokens[3]);
            data.p.y = parse(tokens[4]);
            data.p.z = parse(tokens[5]);

            // 根据数据格式处理不同字段
            if (has_left_eye_data) {
                // 左眼数据字段
                data.ol.x = parse(tokens[6]);
                data.ol.y = parse(tokens[7]);
                data.ol.z = parse(tokens[8]);
                data.pl.x = parse(tokens[9]);
                data.pl.y = parse(tokens[10]);
                data.pl.z = parse(tokens[11]);
                data.F.x = parse(tokens[12]);
                data.F.y = parse(tokens[13]);
                data.F.z = parse(tokens[14]);
                data.E.x = parse(tokens[15]);
                data.E.y = parse(tokens[16]);
                data.E.z = parse(tokens[17]);
            } else {
                // 仅右眼数据
                data.F.x = parse(tokens[6]);
                data.F.y = parse(tokens[7]);
                data.F.z = parse(tokens[8]);
                data.E.x = parse(tokens[9]);
                data.E.y = parse(tokens[10]);
                data.E.z = parse(tokens[11]);
                
                // 左眼字段保持默认值
                data.ol = Geometry_Point();
                data.pl = Geometry_Point();
            }

            data_list.push_back(data);
        } 
        catch (const std::exception& e) {
            std::cerr << "[第" << line_num << "行错误] " << e.what() 
                     << "，已跳过该行" << std::endl;
        }
    }
    return data_list;
}

//////////////////////////////////Aruco二维码检测///////////////////////


void GeometryDetectAruco(cv::Mat &frame, 
                    const  cv::Mat &cameraMatrix,
                    const  cv::Mat &distCoeffs,
                    cv::Mat &R,
                    cv::Mat &tvec) 
{
    // 静态滤波器保持连续帧状态
    static PoseFilter filter;
    
    /*屏幕像素尺寸为1920*1080，物理尺寸为537*296 mm */
	//double w_physical = 527/1920;//单位像素宽度的物理尺寸
    // 物理参数校验
    const float marker_length = 82.34375;    // 单位：毫米,300像素
    const float marker_separation = 32.9375;            //120像素，单位毫米
    const float board_width=2*(marker_length+marker_separation)-marker_separation;
    cv::Mat center_offset = (cv::Mat_<double>(3, 1) << 0, -board_width/2, -marker_separation);
    
    // 创建Board配置
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
                             2, 2, marker_length, marker_separation, dictionary);

    // 增强版检测参数
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    // params->minMarkerDistanceRate = 0.04;       // 提高间距检测要求
    // params->markerBorderBits = 2;               // 提高边界识别精度
    // params->perspectiveRemovePixelPerCell = 8;  // 提高透视校正精度

    // 执行检测
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(frame, dictionary, corners, markerIds, params);

    // 新增有效性校验标志
    bool valid_pose = false;
    if (!markerIds.empty()) {
        cv::Mat rvec;
        int num_valid = cv::aruco::estimatePoseBoard(corners, markerIds, board, 
                                                   cameraMatrix, distCoeffs,
                                                   rvec, tvec);

        // 增强校验条件：至少检测到3个标记且旋转矩阵有效
        if (num_valid >= 3 && !rvec.empty()) {
            cv::Rodrigues(rvec, R);
            //计算中心偏移,原标定板坐标系原点为第3个aruco二维码右下角，现在偏移到中心点
            cv::Mat offset_cam=R*center_offset;
            tvec=tvec-offset_cam;
            
            // 新增姿态突变检测
            if (filter.is_initialized) {
                double rot_diff = cv::norm(R - filter.prevR, cv::NORM_L2);
                double trans_diff = cv::norm(tvec - filter.prevT, cv::NORM_L2);
                
                // 若姿态变化超过阈值则重置滤波器
                if (rot_diff > 0.5 || trans_diff > 50) { // 旋转阈值0.5弧度，平移阈值50mm
                    filter.is_initialized = false;
                }
            }

            // 应用指数加权平均滤波
            if (filter.is_initialized) {
                R = filter.alpha * R + (1 - filter.alpha) * filter.prevR;
                tvec = filter.alpha * tvec + (1 - filter.alpha) * filter.prevT;
            } else {
                filter.is_initialized = true;
            }

            // 更新滤波器状态
            R.copyTo(filter.prevR);
            tvec.copyTo(filter.prevT);
            valid_pose = true;

            // 可视化增强
            cv::aruco::drawDetectedMarkers(frame, corners, markerIds);
            cv::drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 10, 1); // 增大坐标系尺寸
            
            // 绘制滤波后的原点
            std::vector<cv::Point3f> origin{ cv::Point3f(0,0,0) };
            std::vector<cv::Point2f> projected_origin;
            cv::projectPoints(origin, rvec, tvec, cameraMatrix, distCoeffs, projected_origin);
            cv::circle(frame, projected_origin[0], 5, cv::Scalar(0,255,255), -1); // 改用黄色标记滤波后原点
        }
    }

    // 处理无效检测
    if (!valid_pose) {
        R.release();
        tvec.release();
        filter.is_initialized = false; // 重置滤波器
    }
}

void DrawCameraAxes(cv::Mat &frame,const cv::Mat &cameraMatrix) {
    cv::Point2f principal_point(
        cameraMatrix.at<double>(0,2),
        cameraMatrix.at<double>(1,2)
    );
    // 绘制十字线标记主点
    cv::line(frame, 
        cv::Point2f(principal_point.x-10, principal_point.y),
        cv::Point2f(principal_point.x+10, principal_point.y),
        cv::Scalar(0,255,0), 2);
    cv::line(frame, 
        cv::Point2f(principal_point.x, principal_point.y-10),
        cv::Point2f(principal_point.x, principal_point.y+10),
        cv::Scalar(0,255,0), 2);
}
//////////////////////////相机重连///////////////////////////

void ReinitCameras(cv::VideoCapture& scene_cam, int scene_id,
                  cv::VideoCapture& eye_cam, int eye_id,
                    cv::VideoCapture& eye_cam_left, int eye_left_id) 
{
    scene_cam.release();
    eye_cam.release();
    eye_cam_left.release();
    scene_cam.open(scene_id);
    eye_cam.open(eye_id);
    eye_cam_left.open(eye_left_id);
    std::cout << "尝试重新连接摄像头..." << std::endl;
}
void ReinitCameras(cv::VideoCapture& scene_cam, int scene_id,
    cv::VideoCapture& eye_cam, int eye_id) 
{
    scene_cam.release();
    eye_cam.release();
    scene_cam.open(scene_id);
    eye_cam.open(eye_id);
    std::cout << "尝试重新连接摄像头..." << std::endl;
}


Geometry_Point EstimateGazePoint(const cv::Mat& frame,
                        const cv::Mat& R_board_to_cam,
                        const cv::Mat& t_board_to_cam,
                        const Geometry_Line& line_in_cam,
                        const cv::Mat& cameraMatrix,
                        const cv::Mat& distCoeffs)
{
    // ===== 将直线转换到标定板坐标系 =====
    // 标定板到相机的变换矩阵: [R|t]
    // 相机到标定板的变换矩阵: [R.t() | -R.t()*t]
    cv::Mat R_cam_to_board = R_board_to_cam.t();
    cv::Mat t_cam_to_board = -R_cam_to_board * t_board_to_cam;

    // 转换直线起点
    cv::Mat start_cam = (cv::Mat_<double>(3,1) << line_in_cam.start.x,
                                                line_in_cam.start.y,
                                                line_in_cam.start.z);
    cv::Mat start_board = R_cam_to_board * start_cam + t_cam_to_board;

    // 转换方向向量（只旋转不平移）
    cv::Mat dir_cam = (cv::Mat_<double>(3,1) << line_in_cam.direction.x,
                                               line_in_cam.direction.y,
                                               line_in_cam.direction.z);
    cv::Mat dir_board = R_cam_to_board * dir_cam;

    // ===== 建立直线参数方程 =====
    // 标定板坐标系中直线方程：P(t) = start_board + dir_board * t
    // 与yz平面（x=0）的交点需满足：start_board.x + dir_board.x * t = 0

    // ===== 求解参数t =====
    const double epsilon = 1e-6; // 浮点误差阈值
    double denominator = dir_board.at<double>(0);

    // 处理方向向量平行于yz平面的情况
    if (std::abs(denominator) < epsilon) {
        // 检查直线是否在yz平面内
        if (std::abs(start_board.at<double>(0)) < epsilon) {
            std::cout << "直线位于yz平面内" << std::endl;
            // 返回直线上任意一点（这里取起点）
            cv::Mat p_cam = R_board_to_cam * start_board + t_board_to_cam;
            return Geometry_Point(p_cam.at<double>(0),
                                 p_cam.at<double>(1),
                                 p_cam.at<double>(2));
        } else {
            std::cerr << "错误：直线与yz平面无交点" << std::endl;
            return Geometry_Point(); // 返回无效点
        }
    }

    double t = -start_board.at<double>(0) / denominator;

    // ===== 计算交点坐标 =====
    cv::Mat p_board = start_board + dir_board * t;
    
    // 转换回相机坐标系
    cv::Mat p_cam = R_board_to_cam * p_board + t_board_to_cam;

    // ===== 投影到图像平面 =====
    std::vector<cv::Point3f> points3D;
    points3D.emplace_back(p_cam.at<double>(0),
                         p_cam.at<double>(1),
                         p_cam.at<double>(2));

    std::vector<cv::Point2f> points2D;
    cv::projectPoints(points3D, 
                     cv::Mat::zeros(3,1,CV_64F),  // rvec（世界坐标系与相机坐标系重合）
                     cv::Mat::zeros(3,1,CV_64F),  // tvec
                     cameraMatrix,
                     distCoeffs,
                     points2D);

    // 绘制红色交点（半径5像素）
    cv::circle(frame, points2D[0], 5, cv::Scalar(0,0,255), -1);

    return Geometry_Point(p_cam.at<double>(0),
                         p_cam.at<double>(1),
                         p_cam.at<double>(2));
}
Geometry_Point EstimateGazePoint(
    const cv::Mat& frame,
    const Geometry_Line& line_in_cam_left,
    const Geometry_Line& line_in_cam_right,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs)
{
    // ===== 输入校验 =====
    if (line_in_cam_left.direction.Magnitude() < 1e-6 || 
        line_in_cam_right.direction.Magnitude() < 1e-6) {
        std::cerr << "错误：无效的视线方向向量" << std::endl;
        return Geometry_Point();
    }

    // ===== 参数准备 =====
    // 左眼线参数方程：P = L0 + t * u
    Geometry_Point L0 = line_in_cam_left.start;
    Geometry_Vector u = line_in_cam_left.direction;

    // 右眼线参数方程：Q = R0 + s * v
    Geometry_Point R0 = line_in_cam_right.start;
    Geometry_Vector v = line_in_cam_right.direction;

    // ===== 计算异面直线最近点 =====
    // 向量w0 = R0 - L0
    Geometry_Vector w0(R0.x - L0.x, R0.y - L0.y, R0.z - L0.z);
    
    // 计算分母项
    double a = u.DotProduct(u);
    double b = u.DotProduct(v);
    double c = v.DotProduct(v);
    double d = u.DotProduct(w0);
    double e = v.DotProduct(w0);
    
    // 解线性方程组求参数t和s
    double denominator = a*c - b*b;
    if (std::abs(denominator) < 1e-6) {
        std::cerr << "警告：视线接近平行，使用中点近似" << std::endl;
        // 取两线起点的中点
        Geometry_Point midpoint(
            (L0.x + R0.x)/2, 
            (L0.y + R0.y)/2,
            (L0.z + R0.z)/2
        );
        return midpoint;
    }

    double t = (b*e - c*d) / denominator;
    double s = (a*e - b*d) / denominator;

    // ===== 计算最近点坐标 =====
    Geometry_Point P = Geometry_Point(
        L0.x + u.x * t,
        L0.y + u.y * t,
        L0.z + u.z * t
    );
    
    Geometry_Point Q = Geometry_Point(
        R0.x + v.x * s,
        R0.y + v.y * s,
        R0.z + v.z * s
    );

    // 取中点作为最终凝视点
    Geometry_Point gaze_point(
        (P.x + Q.x)/2,
        (P.y + Q.y)/2,
        (P.z + Q.z)/2
    );

    // ===== 可视化投影 =====
    std::vector<cv::Point3f> points3D;
    points3D.emplace_back(gaze_point.x, gaze_point.y, gaze_point.z);
    points3D.emplace_back(P.x, P.y, P.z);    // 左视线最近点
    points3D.emplace_back(Q.x, Q.y, Q.z);    // 右视线最近点

    std::vector<cv::Point2f> points2D;
    cv::projectPoints(points3D, 
        cv::Mat::zeros(3,1,CV_64F), 
        cv::Mat::zeros(3,1,CV_64F),
        cameraMatrix,
        distCoeffs,
        points2D);

    // 绘制凝视点（绿色）
    cv::circle(frame, points2D[0], 8, cv::Scalar(0,255,0), -1);
    // 绘制左视线最近点（蓝色）
    cv::circle(frame, points2D[1], 5, cv::Scalar(255,0,0), -1);
    // 绘制右视线最近点（红色）
    cv::circle(frame, points2D[2], 5, cv::Scalar(0,0,255), -1);

    return gaze_point;
}


