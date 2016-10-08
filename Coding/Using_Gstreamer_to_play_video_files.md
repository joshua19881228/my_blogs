# Gstreamer #

The official website of Gstreamer is [here](https://gstreamer.freedesktop.org/). What is Gstreamer is quoted from its website:

>GStreamer is a library for constructing graphs of media-handling components. The applications it supports range from simple Ogg/Vorbis playback, audio/video streaming to complex audio (mixing) and video (non-linear editing) processing.
>
>Applications can take advantage of advances in codec and filter technology transparently. Developers can add new codecs and filters by writing a simple plugin with a clean, generic interface.
>
>GStreamer is released under the LGPL. The 1.x series is API and ABI stable and supersedes the previous stable 0.10 series. Both can be installed in parallel.

# A Simple Tutorial #

I started to learn to use Gstreamer just two days ago, and I found a very useful tutorial called [Gstreamer Small Tutorial](https://arashafiei.files.wordpress.com/2012/12/gst-doc.pdf) authored by [Arash Shafiei](https://github.com/ashafiei). One could use this tutorial as a stepping-stone to develop more complex applications. A more detailed tutorial can be found in Gstreamer's [website](http://docs.gstreamer.com/display/GstSDK/Tutorials).

# My Own Trial #

Though Arash Shafiei provided an excellent sample, there is some modifications that need to be done to run the application to play video correctly.

1. more recent API should be used by replacing `gst_pad_get_caps` with `gst_pad_query_caps`.
2. there's a mistake in Arash Shafiei's sample code of `static void pad_added_handler(GstElement *src, GstPad *new_pad, CustomData *data)`. The original code will not link video sink if the audio sink has already linked no matter whether video sink is linked or not.
3. a demuxer is added to the pipeline to handle the input of a file source element.

my own code can be found following:

    #include <gst/gst.h>
    #include <glib.h>
    /* Structure to contain all our information, so we can pass it to callbacks */
    typedef struct _CustomData
    {
        GstElement *pipeline;
        GstElement *source;
        GstElement *demuxer;
        GstElement *video_convert;
        GstElement *audio_convert;
        GstElement *video_sink;
        GstElement *audio_sink;
    } CustomData;
    /* Handler for the pad-added signal */
    /* This function will be called by the pad-added signal */
    static void pad_added_handler(GstElement *src, GstPad *new_pad, CustomData *data)
    {
        GstPad *sink_pad_audio = gst_element_get_static_pad(data->audio_convert, "sink");
        GstPad *sink_pad_video = gst_element_get_static_pad(data->video_convert, "sink");
        GstPadLinkReturn ret;
        GstCaps *new_pad_caps = NULL;
        GstStructure *new_pad_struct = NULL;
        const gchar *new_pad_type = NULL;
        g_print("Received new pad '%s' from '%s':\n", GST_PAD_NAME(new_pad), GST_ELEMENT_NAME(src));

        /* Check the new pad's type */
        new_pad_caps = gst_pad_query_caps(new_pad, 0);
        new_pad_struct = gst_caps_get_structure(new_pad_caps, 0);
        new_pad_type = gst_structure_get_name(new_pad_struct);
        if (g_str_has_prefix(new_pad_type, "audio/x-raw"))
        {
            /* If our audio converter is already linked, we have nothing to do here */
            if (gst_pad_is_linked(sink_pad_audio))
            {
                g_print(" Type is '%s'.\n", new_pad_type);
                g_print(" We are already linked. Ignoring.\n");
                goto exit;
            }
            /* Attempt the link */
            ret = gst_pad_link(new_pad, sink_pad_audio);
            if (GST_PAD_LINK_FAILED(ret))
            {
                g_print(" Type is '%s' but link failed.\n", new_pad_type);
            }
            else
            {
                g_print(" Link succeeded (type '%s').\n", new_pad_type);
            }
        }
        else if (g_str_has_prefix(new_pad_type, "video/x-raw"))
        {
            /* If our video converter is already linked, we have nothing to do here */
            if (gst_pad_is_linked(sink_pad_video))
            {
                g_print(" Type is '%s'.\n", new_pad_type);
                g_print(" We are already linked. Ignoring.\n");
                goto exit;
            }
            /* Attempt the link */
            ret = gst_pad_link(new_pad, sink_pad_video);
            if (GST_PAD_LINK_FAILED(ret))
            {
                g_print(" Type is '%s' but link failed.\n", new_pad_type);
            }
            else
            {
                g_print(" Link succeeded (type '%s').\n", new_pad_type);
            }
        }
        else
        {
            g_print(" It has type '%s' which is not raw audio. Ignoring.\n", new_pad_type);
            goto exit;
        }
    exit:
        /* Unreference the new pad's caps, if we got them */
        if (new_pad_caps != NULL)
            gst_caps_unref(new_pad_caps);
        /* Unreference the sink pad */
        gst_object_unref(sink_pad_audio);
        gst_object_unref(sink_pad_video);
    }

    int main(int argc, char *argv[])
    {

        if(argc != 2)
        {
            g_printerr("usage: ./player <path_to_a_video>\n");
            return 0;
        }

        CustomData data;
        GstBus *bus;
        GstMessage *msg;
        GstStateChangeReturn ret;
        gboolean terminate = FALSE;
        /* Initialize GStreamer */
        gst_init(&argc, &argv);
        /* Create the elements */
        data.source = gst_element_factory_make("filesrc", "source");
        data.demuxer = gst_element_factory_make("decodebin", "demuxer");
        data.audio_convert = gst_element_factory_make("audioconvert", "audio_convert");
        data.audio_sink = gst_element_factory_make("autoaudiosink", "audio_sink");
        data.video_convert = gst_element_factory_make("videoconvert", "video_convert");
        data.video_sink = gst_element_factory_make("autovideosink", "video_sink");
        /* Create the empty pipeline */
        data.pipeline = gst_pipeline_new("test-pipeline");
        if (!data.pipeline || !data.source || !data.audio_convert ||
            !data.audio_sink || !data.video_convert || !data.video_sink)
        {
            g_printerr("Not all elements could be created.\n");
            return -1;
        }
        /* Build the pipeline. Note that we are NOT linking the source at this point. We will do it later. */
        gst_bin_add_many(GST_BIN(data.pipeline), data.source, data.demuxer,
                        data.audio_convert, data.audio_sink, data.video_convert, data.video_sink, NULL);
        if (!gst_element_link(data.source, data.demuxer))
        {
            g_printerr("Elements could not be linked.\n");
            gst_object_unref(data.pipeline);
            return -1;
        }
        if (!gst_element_link(data.audio_convert, data.audio_sink))
        {
            g_printerr("Elements could not be linked.\n");
            gst_object_unref(data.pipeline);
            return -1;
        }
        if (!gst_element_link(data.video_convert, data.video_sink))
        {
            g_printerr("Elements could not be linked.\n");
            gst_object_unref(data.pipeline);
            return -1;
        }
        /* Set the file to play */
        g_object_set(data.source, "location", argv[1], NULL);
        /* Connect to the pad-added signal */
        g_signal_connect(data.demuxer, "pad-added", G_CALLBACK(pad_added_handler), &data);
        /* Start playing */
        ret = gst_element_set_state(data.pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE)
        {
            g_printerr("Unable to set the pipeline to the playing state.\n");
            gst_object_unref(data.pipeline);
            return -1;
        }
        /* Listen to the bus */
        bus = gst_element_get_bus(data.pipeline);
        do
        {
            msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
            (GstMessageType)(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            /* Parse message */
            if (msg != NULL)
            {
                GError *err;
                gchar *debug_info;
                switch (GST_MESSAGE_TYPE(msg))
                {
                case GST_MESSAGE_ERROR:
                    gst_message_parse_error(msg, &err, &debug_info);
                    g_printerr("Error received from element %s: %s\n", GST_OBJECT_NAME(msg->src), err->message);
                    g_printerr("Debugging information: %s\n", debug_info ? debug_info : "none");
                    g_clear_error(&err);
                    g_free(debug_info);
                    terminate = TRUE;
                    break;
                case GST_MESSAGE_EOS:
                    g_print("End-Of-Stream reached.\n");
                    terminate = TRUE;
                    break;
                case GST_MESSAGE_STATE_CHANGED:
                    /* We are only interested in state-changed messages from the pipeline */
                    if (GST_MESSAGE_SRC(msg) == GST_OBJECT(data.pipeline))
                    {
                        GstState old_state, new_state, pending_state;
                    gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                    g_print("Pipeline state changed from %s to %s:\n",
                            gst_element_state_get_name(old_state), gst_element_state_get_name(new_state));
                }
                break;
            default:
                /* We should not reach here */
                g_printerr("Unexpected message received.\n");
                break;
                }
            gst_message_unref(msg);
            }
        } while (!terminate);
        /* Free resources */
        gst_object_unref(bus);
        gst_element_set_state(data.pipeline, GST_STATE_NULL);
        gst_object_unref(data.pipeline);
        return 0;
    }

the compiling commond is 

    gcc player.c -o player `pkg-config --cflags --libs gstreamer-1.0`
