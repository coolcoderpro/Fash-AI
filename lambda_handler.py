
import json

import base64

from query_pipeline import run_query, run_match_only, run_text_match_only, run_compare_insights, run_text_search



def handler(event, context):

    import boto3

    sts = boto3.client("sts")

    identity = sts.get_caller_identity()

    print("LAMBDA IDENTITY:", identity["Arn"])



    if event.get("httpMethod") == "OPTIONS":

        return {"statusCode": 200, "headers": cors_headers(), "body": ""}



    try:

        body = json.loads(event.get("body", "{}"))

        mode = body.get("mode", "default")



        if mode == "compare_insights":

            result = run_compare_insights(

                body.get("products1", []),

                body.get("products2", []),

                body.get("desc1", ""),

                body.get("desc2", "")

            )

            return {"statusCode": 200, "headers": cors_headers(), "body": json.dumps(result)}



        image_b64   = body.get("image")

        description = body.get("description", "")



        if mode == "match_only":

            if image_b64:

                image_bytes = base64.b64decode(image_b64)

                result = run_match_only(image_bytes, description)

            elif description:

                result = run_text_match_only(description)

            else:

                return {"statusCode": 400, "headers": cors_headers(), "body": json.dumps({"error": "No image or description provided"})}

            return {"statusCode": 200, "headers": cors_headers(), "body": json.dumps(result)}



        if not image_b64 and description:

            result = run_text_search(description)

            return {"statusCode": 200, "headers": cors_headers(), "body": json.dumps(result)}



        if not image_b64:

            return {"statusCode": 400, "headers": cors_headers(), "body": json.dumps({"error": "No image or description provided"})}



        image_bytes = base64.b64decode(image_b64)

        result = run_query(image_bytes, description)



        return {"statusCode": 200, "headers": cors_headers(), "body": json.dumps(result)}



    except Exception as e:

        print("ERROR:", str(e))

        import traceback; traceback.print_exc()

        return {"statusCode": 500, "headers": cors_headers(), "body": json.dumps({"error": str(e)})}



def cors_headers():

    return {

        "Content-Type":                "application/json",

        "Access-Control-Allow-Origin": "*",

        "Access-Control-Allow-Headers": "Content-Type",

        "Access-Control-Allow-Methods": "POST,OPTIONS"

    }

