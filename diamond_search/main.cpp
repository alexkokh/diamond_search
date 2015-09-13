#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;

enum ME_Precision { FullPel = 0, HalfPel, QPel };
enum SearchPattern { LDSP = 0, SDSP };

int32_t LDSP_offset[9] = {0};
int32_t SDSP_offset[5] = {0};
int32_t LDSP_x[9] = {0};
int32_t LDSP_y[9] = {0};
int32_t SDSP_x[5] = {0};
int32_t SDSP_y[5] = {0};

void calc_ds_offsets(const int32_t width, const int32_t block_size)
{
    LDSP_offset[0] = 0;             //       5
    LDSP_offset[1] = -2;            //     3   4
    LDSP_offset[2] = 2;             //   1   0   2
    LDSP_offset[3] = -(width+1);    //     6   7
    LDSP_offset[4] = -(width-1);    //       8
    LDSP_offset[5] = -2*width;
    LDSP_offset[6] = width-1;
    LDSP_offset[7] = width+1;
    LDSP_offset[8] = 2*width;
    
    LDSP_x[0] = 0;
    LDSP_x[1] = -2;
    LDSP_x[2] = 2;
    LDSP_x[3] = -1;
    LDSP_x[4] = 1;
    LDSP_x[5] = 0;
    LDSP_x[6] = -1;
    LDSP_x[7] = 1;
    LDSP_x[8] = 0;
    
    LDSP_y[0] = 0;
    LDSP_y[1] = 0;
    LDSP_y[2] = 0;
    LDSP_y[3] = -1;
    LDSP_y[4] = -1;
    LDSP_y[5] = -2;
    LDSP_y[6] = 1;
    LDSP_y[7] = 1;
    LDSP_y[8] = 2;
    
    SDSP_offset[0] = 0;            //       3
    SDSP_offset[1] = -1;           //     1 0 2
    SDSP_offset[2] = 1;            //       4
    SDSP_offset[3] = -width;
    SDSP_offset[4] = width;
    
    SDSP_x[0] = 0;
    SDSP_x[1] = -1;
    SDSP_x[2] = 1;
    SDSP_x[3] = 0;
    SDSP_x[4] = 0;

    SDSP_y[0] = 0;
    SDSP_y[1] = 0;
    SDSP_y[2] = 0;
    SDSP_y[3] = -1;
    SDSP_y[4] = 1;
}

typedef struct mv_s
{
    int32_t x;
    int32_t y;
} mv_t;

mv_t *mvs;

uint32_t MSE(const uint8_t* cur, const uint8_t* prev, const int32_t block_size, const int32_t stride)
{
    uint32_t block_MSE = 0;
    
    for(int32_t i = 0; i < block_size; i++)
    {
        for(int32_t j = 0; j < block_size; j++)
        {
            int32_t dif = cur[j] - prev[j];
            block_MSE += dif*dif;
        }
        
        cur += stride;
        prev += stride;
    }
    
    return block_MSE;
}

uint32_t get_min_MSE(const uint32_t* const blocks, const int32_t num_blocks)
{
    uint32_t min_MSE_block_id = 0;
    
    for(int32_t i = 1; i < num_blocks; i++)
    {
        if(blocks[i] < blocks[min_MSE_block_id])
            min_MSE_block_id = i;
    }
    
    return min_MSE_block_id;
}

uint32_t get_min_block(const Mat& cur, const Mat& prev,
                       const int32_t x_cur, const int32_t y_cur,
                       const int32_t x_prev, const int32_t y_prev,
                       const int32_t block_size,
                       const SearchPattern mode)
{
    uint32_t block_MSE[9] = {0};
    uint32_t num_blocks;
    int32_t *offset;
    int32_t *offset_x;
    int32_t *offset_y;
    
    const int32_t cur_offset = y_cur*cur.cols + x_cur;
    const int32_t start_prev_offset = y_prev*cur.cols + x_prev;
    int32_t prev_offset;
    
    if(mode == LDSP)
    {
        num_blocks = 9;
        offset = LDSP_offset;
        offset_x = LDSP_x;
        offset_y = LDSP_y;
    } else
    {
        num_blocks = 5;
        offset = SDSP_offset;
        offset_x = SDSP_x;
        offset_y = SDSP_y;
    }
    
    for(int32_t i = 0; i < num_blocks; i++)
    {
        prev_offset = start_prev_offset + offset[i];
        
        if(x_prev+offset_x[i] < 0 || y_prev+offset_y[i] < 0 ||
           x_prev+offset_x[i] + block_size > cur.cols-1 || y_prev+offset_y[i] + block_size > cur.rows-1)
        {
            block_MSE[i] = INT32_MAX;
        } else
        {
            block_MSE[i] = MSE(cur.data + cur_offset, prev.data + prev_offset, block_size, cur.cols);
        }
    }

    return get_min_MSE(block_MSE, num_blocks);
}

mv_t diamond_search(const Mat& cur, const Mat& prev, const int32_t x, int32_t y, const int32_t block_size)
{
    mv_t mv {0,0};
    uint32_t min_LDSP_block;
    uint32_t min_SDSP_block;
    const int32_t xc = x;
    const int32_t yc = y;
    int32_t xp = x;
    int32_t yp = y;
    
    do
    {
        min_LDSP_block = get_min_block(cur, prev, xc, yc, xp, yp, block_size, LDSP);
        xp += LDSP_x[min_LDSP_block];
        yp += LDSP_y[min_LDSP_block];
    } while(min_LDSP_block);
    
    min_SDSP_block = get_min_block(cur, prev, xc, yc, xp, yp, block_size, SDSP);
    
    xp += SDSP_x[min_SDSP_block];
    yp += SDSP_y[min_SDSP_block];
    
    mv.x = xp - xc;
    mv.y = yp - yc;
    
    return mv;
}

void motion_estimation(const Mat& cur, const Mat& prev, const uint32_t block_size, const ME_Precision mode)
{
    assert(!(block_size % 2));
    assert(mode == FullPel); // only full pel now
    
    mv_t *mv_buf = mvs;
    
    for(int i = 0, n = 0; i < cur.rows; i+=block_size)
    {
        for (int j = 0; j < cur.cols; j+=block_size)
        {
            mv_buf[n++] = diamond_search(cur, prev, j, i, block_size);
        }
    }
}

void draw_mvs(Mat& frame, const mv_t* const mvs, const uint32_t block_size)
{
    for(int i = 0, n = 0; i < frame.rows; i+=block_size)
    {
        for (int j = 0; j < frame.cols; j+=block_size)
        {
            mv_t mv = mvs[n++];
            Point pt0, pt1;
            pt0.x = j + block_size/2;
            pt0.y = i + block_size/2;
            
            pt1.x = pt0.x + mv.x;
            pt1.y = pt0.y + mv.y;
            
            line(frame, pt0, pt1, Scalar(0,0,255));
        }
    }
}

int main()
{
    const int32_t block_size = 16;
    Mat frame[2];
    Mat cur, prev, out;
    
	VideoCapture cap(0);

	if (!cap.isOpened())

		return -1;

	namedWindow("me", 1);

    cap.read(frame[0]);
    
    assert(frame[0].cols == frame[0].cols);
    assert(frame[0].rows == frame[0].rows);
    assert(!(frame[0].cols % block_size));
    assert(!(frame[0].rows % block_size));
    
    calc_ds_offsets(frame[0].cols, block_size);

    const uint32_t width_blocks = frame[0].cols / block_size;
    const uint32_t height_blocks = frame[0].rows / block_size;
    
    mvs = (mv_t *)malloc(width_blocks * height_blocks * sizeof(mv_t));

    int32_t cur_id = 1;
    int32_t prev_id = 0;

	for (;;)
	{
		long l1, l2;

		l1 = clock();
		cap.read(frame[cur_id]);
        
        cvtColor(frame[prev_id], prev, CV_BGR2GRAY);
        cvtColor(frame[cur_id], cur, CV_BGR2GRAY);
        
        motion_estimation(cur, prev, block_size, FullPel);
        
        frame[cur_id].copyTo(out);
        draw_mvs(out, mvs, block_size);

		l2 = clock();

		char diag[1024] = { 0 };
		sprintf(diag, "%i FPS", (int)(CLOCKS_PER_SEC/(l2-l1)));
		putText(out, diag, Point(5, frame[cur_id].rows - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 40), 2);

		imshow("me", out);
        
        prev_id = 1 - prev_id;
        cur_id = 1 - cur_id;
        
		if (waitKey(10) >= 0) break;
	}
    
    free(mvs);

	return 0;
}