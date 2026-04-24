#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>

class MeshPublisherNode : public rclcpp::Node
{
public:
    MeshPublisherNode() : Node("mesh_publisher")
    {
        this->declare_parameter<std::string>(
            "mesh_file", 
            "file:///home/vlak/lib/habitat-sim/data/scene_datasets/habitat-test-scenes/skokloster-cast.glb"
        );

        rclcpp::QoS qos_profile(1);
        qos_profile.transient_local();

        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("static_mesh", qos_profile);

        publish_mesh();
    }

private:
    void publish_mesh()
    {
        auto marker = visualization_msgs::msg::Marker();

        marker.header.frame_id = "world"; 
        marker.header.stamp.sec = 0;
        marker.header.stamp.nanosec = 0;
        marker.ns = "mesh";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.mesh_resource = this->get_parameter("mesh_file").as_string();
        marker.mesh_use_embedded_materials = true;

        marker.scale.x = 1.0;
        marker.scale.y = 1.0;
        marker.scale.z = 1.0;

        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        marker.pose.position.x = 0.0;
        marker.pose.position.y = 0.0;
        marker.pose.position.z = 0.0;
        marker.pose.orientation.w = 1.0;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;

        marker_pub_->publish(marker);
        RCLCPP_INFO(this->get_logger(), "Mesh published! Resource: %s", marker.mesh_resource.c_str());
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MeshPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}