import org.springframework.web.client.RestTemplate;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpMethod;
import org.json.JSONObject;

public class PythonApiClient {
    public static void main(String[] args) {
        // FastAPI 서버 URL
        String url = "http://127.0.0.1:8000/predict/";

        // 요청 JSON 데이터 생성
        JSONObject requestJson = new JSONObject();
        requestJson.put("user_id", 101);
        requestJson.put("total_mission_count", 50);
        requestJson.put("total_clicks", 300);

        // HTTP 요청 헤더 설정
        HttpHeaders headers = new HttpHeaders();
        headers.set("Content-Type", "application/json");

        // HTTP 요청 객체 생성
        HttpEntity<String> request = new HttpEntity<>(requestJson.toString(), headers);

        // RestTemplate을 사용하여 API 호출
        RestTemplate restTemplate = new RestTemplate();
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, request, String.class);

        // 응답 출력
        System.out.println("Python API 응답: " + response.getBody());
    }
}